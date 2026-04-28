from datetime import datetime
import time
import shutil
import torch
import torch._dynamo
import torch.optim as optim
import numpy as np
from collections import deque
import os

from environment.display import print_colored

# Default hyperparameters (can be overridden by config)
DROPOUT_RATE = 0.2
WEIGHT_INIT = "xavier_uniform"

from .networks import ActorNetwork, CriticNetwork, QNetwork, pretensorize_actor_observations, pretensorize_critic_observations


class MAPPOAgent:
    def __init__(
        self,
        env,
        hidden_size=256,
        lr_actor=0.0001,
        lr_critic=0.0001,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.1,
        batch_size=32768,
        buffer_size=None,
        num_epochs=10,
        dropout_rate=None,
        weight_init=None,
        device=None,
        compile_models=False,
        communication_config=None,
        volunteer_threshold=None,
        use_coma=True,
        gate_in_ratio=False,
        gate_entropy_coef=0.5,
    ):
        self.dropout_rate = dropout_rate if dropout_rate is not None else DROPOUT_RATE
        self.weight_init = weight_init if weight_init is not None else WEIGHT_INIT
        self.volunteer_threshold = volunteer_threshold  # If set, use threshold instead of argmax for deterministic
        self.use_coma = use_coma
        self.gate_in_ratio = gate_in_ratio  # Include gate log-probs in PPO importance ratio
        self.gate_entropy_coef = gate_entropy_coef  # Gate entropy loss coefficient
        self.env = env
        self.n_agents = len(env.agents)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.num_epochs = num_epochs

        # Set device for GPU/MPS acceleration
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                # Enable TensorFloat32 for faster matmul on Ampere+ GPUs (A100, H100, H200)
                # This uses TF32 tensor cores: ~2x faster, precision is still sufficient for RL
                torch.set_float32_matmul_precision('high')
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        print_colored(f"MAPPOAgent using device: {self.device}", "blue")

        # Create actor networks for each agent
        self.actors = {}
        for agent in env.agents:
            obs_space = env.observation_space(agent.id)
            action_space = env.action_space(agent.id)
            self.actors[agent.id] = ActorNetwork(
                obs_space,
                action_space,
                hidden_size=hidden_size,
                dropout_rate=self.dropout_rate,
                weight_init=self.weight_init,
                device=self.device,
            ).to(self.device)

        # Create centralized critic (V-network for GAE)
        first_agent = env.agents[0]
        self.critic = CriticNetwork(
            env.observation_space(first_agent.id),
            self.n_agents,
            hidden_size=2 * hidden_size,
            dropout_rate=self.dropout_rate,
            weight_init=self.weight_init,
            device=self.device,
        ).to(self.device)

        # Create COMA Q-network for counterfactual baselines (only if COMA enabled)
        if self.use_coma:
            self.q_network = QNetwork(
                env.observation_space(first_agent.id),
                self.n_agents,
                n_actions=2,  # volunteer or not
                hidden_size=2 * hidden_size,
                dropout_rate=self.dropout_rate,
                weight_init=self.weight_init,
                device=self.device,
            ).to(self.device)
        else:
            self.q_network = None

        # Setup optimizers
        self.actor_optimizers = {
            agent_id: optim.Adam(actor.parameters(), lr=lr_actor)
            for agent_id, actor in self.actors.items()
        }
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=lr_critic) if self.use_coma else None

        # === Communication Setup (IC3Net / MLP-Comm) ===
        self.use_ic3net = False  # Flag for any comm-actor variant (IC3Net or MLP-Comm)
        if communication_config and communication_config.get('enabled', False):
            comm_type = communication_config.get('type', '')
            if comm_type in ('ic3net', 'mlp_comm'):
                comm_disabled_flag = communication_config.get('comm_disabled', False)
                comm_status = "DISABLED (no-comm baseline)" if comm_disabled_flag else "ENABLED"
                backbone_label = "LSTM (IC3Net)" if comm_type == 'ic3net' else "MLP"

                print_colored(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Initializing {backbone_label} comm actor "
                    f"(hidden_size={communication_config.get('hidden_size', 64)}, "
                    f"comm_rounds={communication_config.get('n_comm_rounds', 2)}, "
                    f"communication={comm_status})",
                    "green",
                )

                self.use_ic3net = True  # Reuse same training path for both
                self.ic3net_comm_disabled = comm_disabled_flag

                # Select the right actor class
                if comm_type == 'ic3net':
                    from .communication import IC3NetActor
                    ActorClass = IC3NetActor
                else:
                    from .communication import MLPCommActor
                    ActorClass = MLPCommActor

                self.ic3net_actor = ActorClass(
                    obs_space=env.observation_space(first_agent.id),
                    action_space=env.action_space(first_agent.id),
                    n_agents=self.n_agents,
                    hidden_size=communication_config.get('hidden_size', 64),
                    n_comm_rounds=communication_config.get('n_comm_rounds', 2),
                    comm_dropout=communication_config.get('comm_dropout', 0.0),
                    weight_init=self.weight_init,
                    comm_disabled=comm_disabled_flag,
                ).to(self.device)

                # Communication warmup: disable comm for first N episodes
                comm_warmup = communication_config.get('comm_warmup_episodes', 0)
                if comm_warmup > 0 and not comm_disabled_flag:
                    self.ic3net_actor.set_comm_warmup(comm_warmup)
                    print_colored(
                        f"[{datetime.now().strftime('%H:%M:%S')}] Comm warmup: communication disabled for first {comm_warmup} episodes",
                        "yellow",
                    )

                # Single shared optimizer (weight sharing)
                self.ic3net_optimizer = optim.Adam(
                    self.ic3net_actor.parameters(), lr=lr_actor
                )
                print_colored(
                    f"[{datetime.now().strftime('%H:%M:%S')}] {backbone_label}: {sum(p.numel() for p in self.ic3net_actor.parameters()):,} parameters (shared across {self.n_agents} agents)",
                    "green",
                )

        # Apply torch.compile() to training methods only (only on CUDA, not MPS)
        # We compile forward_batch_pretensorized (not the full model) because:
        # - forward() is called during CPU rollout with varying Python objects → dynamo recompilation
        # - forward_batch_pretensorized() is called during GPU training with consistent tensor shapes → compiles once
        self._compiled = False
        if compile_models and self.device.type == "cuda" and hasattr(torch, "compile"):
            print_colored(
                f"[{datetime.now().strftime('%H:%M:%S')}] Compiling forward_batch_pretensorized methods with torch.compile...",
                "cyan",
            )
            try:
                # Enable suppress_errors so any runtime compilation failures
                # (e.g. corrupted inductor cache) gracefully fall back to eager mode
                torch._dynamo.config.suppress_errors = True

                # Clear corrupted inductor cache if it exists
                cache_dir = os.path.join("/tmp", f"torchinductor_{os.environ.get('USER', 'root')}")
                if os.path.exists(cache_dir):
                    try:
                        shutil.rmtree(cache_dir)
                        print_colored(
                            f"[{datetime.now().strftime('%H:%M:%S')}] Cleared TorchInductor cache: {cache_dir}",
                            "cyan",
                        )
                    except Exception:
                        pass  # Not critical if cache can't be cleared

                # Compile only the pretensorized forward methods (used during GPU training)
                for actor in self.actors.values():
                    actor.forward_batch_pretensorized = torch.compile(
                        actor.forward_batch_pretensorized, mode="default"
                    )
                self.critic.forward_batch_pretensorized = torch.compile(
                    self.critic.forward_batch_pretensorized, mode="default"
                )
                # Compile IC3Net if enabled
                if self.use_ic3net:
                    self.ic3net_actor.forward_batch_pretensorized = torch.compile(
                        self.ic3net_actor.forward_batch_pretensorized, mode="default"
                    )
                self._compiled = True
                print_colored(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Compilation complete (training methods only, suppress_errors=True)",
                    "green",
                )
            except Exception as e:
                print_colored(
                    f"[{datetime.now().strftime('%H:%M:%S')}] torch.compile failed, falling back to eager mode: {e}",
                    "yellow",
                )
        elif compile_models and self.device.type == "mps":
            print_colored(
                f"[{datetime.now().strftime('%H:%M:%S')}] torch.compile skipped (not supported on MPS)",
                "yellow",
            )

        # Initialize models on CPU for simulation (will be moved to device during training)
        self._models_on_cpu = False  # Track current location of models
        self._move_models_to_cpu()  # Start with models on CPU for simulation

        # Experience buffer
        self.buffer = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "values": [],
            "action_probs": [],
            "advantages": [],
            "returns": [],
        }
        if self.use_ic3net:
            self.buffer["gate_log_probs"] = []
            self.buffer["gate_actions"] = []

        # For tracking training performance
        self.episode_rewards = deque(maxlen=100)

    def select_actions(self, observations, deterministic=False):
        """Select actions for all agents based on their observations.

        Applies invalid action masking: agents that cannot perform the upcoming
        task have their action forced to 0 (don't volunteer). This is standard
        practice in MARL (Huang & Ontañón 2020, Yu et al. 2022) and prevents
        the network from wasting capacity learning trivially invalid actions.
        """
        # Ensure models are on CPU for inference during episode collection
        self._move_models_to_cpu()

        if self.use_ic3net:
            # IC3Net path: all agents communicate before selecting actions
            with torch.no_grad():
                reward = self.buffer["rewards"][-1] if self.buffer["rewards"] else None
                actions, action_probs, gate_log_probs, gate_actions = (
                    self.ic3net_actor.act(
                        observations, reward, deterministic,
                        volunteer_threshold=self.volunteer_threshold if deterministic else None,
                    )
                )
                # Store gate info internally for store_experience() to pick up
                self.ic3net_actor._last_gate_log_probs = gate_log_probs
                self.ic3net_actor._last_gate_actions = gate_actions

            # Apply action masking for non-capable agents
            for agent_id, obs in observations.items():
                if obs["agent_is_capable"] == 0:
                    actions[agent_id] = 0
                    # Set probs to [1.0, 0.0] (certain don't-volunteer)
                    masked_probs = torch.zeros_like(action_probs[agent_id])
                    masked_probs[0] = 1.0
                    action_probs[agent_id] = masked_probs

            return actions, action_probs
        else:
            # Standard MAPPO path: independent actors
            actions = {}
            action_probs = {}
            with torch.no_grad():
                for agent_id, obs in observations.items():
                    if agent_id in self.actors:
                        last_reward = self.buffer["rewards"][-1] if self.buffer["rewards"] else None
                        # Extract per-agent reward if dict, else use as-is
                        if isinstance(last_reward, dict):
                            reward = last_reward.get(agent_id, 0.0)
                        else:
                            reward = last_reward
                        action, probs = self.actors[agent_id].act(
                            obs, reward, deterministic,
                            volunteer_threshold=self.volunteer_threshold if deterministic else None,
                        )

                        # Action masking: force non-capable agents to not volunteer
                        if obs["agent_is_capable"] == 0:
                            action = 0
                            probs = torch.zeros_like(probs)
                            probs[0] = 1.0  # Certain don't-volunteer

                        actions[agent_id] = action
                        action_probs[agent_id] = probs
            return actions, action_probs

    def compute_values(self, observations):
        """Compute values for the current observations using the critic network."""
        # Convert observations dict to list in agent order
        obs_list = [observations[agent.id] for agent in self.env.agents]
        # Get the reward from the last step
        reward = self.buffer["rewards"][-1] if self.buffer["rewards"] else None

        # Ensure critic is on CPU for inference during episode collection
        self._move_models_to_cpu()

        with torch.no_grad():
            value = self.critic(
                obs_list, reward
            ).item()  # Already on CPU since model is on CPU
        return value

    def _compute_values_on_current_device(self, observations):
        """Compute values using the critic network on its current device (for use during training)."""
        # Convert observations dict to list in agent order
        obs_list = [observations[agent.id] for agent in self.env.agents]
        # Get the reward from the last step
        reward = self.buffer["rewards"][-1] if self.buffer["rewards"] else None

        with torch.no_grad():
            value = self.critic(obs_list, reward).item()
        return value

    def store_experience(self, obs, actions, action_probs, rewards, dones, values):
        """Store experience in the buffer."""
        self.buffer["obs"].append(obs)
        self.buffer["actions"].append(actions)
        self.buffer["action_probs"].append(action_probs)
        self.buffer["rewards"].append(rewards)
        self.buffer["dones"].append(dones)
        self.buffer["values"].append(values)

        # Automatically pick up IC3Net gate data if available
        if self.use_ic3net and self.ic3net_actor._last_gate_log_probs is not None:
            self.buffer["gate_log_probs"].append(
                self.ic3net_actor._last_gate_log_probs.clone()
            )
            self.buffer["gate_actions"].append(
                self.ic3net_actor._last_gate_actions.clone()
            )

    def reset_history(self):
        """Reset the history buffers for all networks."""
        for actor in self.actors.values():
            actor.reset_history()
        self.critic.reset_history()
        if self.use_ic3net:
            self.ic3net_actor.reset_hidden_states()

    def set_current_episode(self, episode: int) -> None:
        """Update episode counter (used for comm warmup scheduling)."""
        if self.use_ic3net and hasattr(self.ic3net_actor, 'set_current_episode'):
            self.ic3net_actor.set_current_episode(episode)

    def compute_advantages_and_returns(self, use_current_device=False):
        """Compute GAE advantages and returns for stored trajectories.

        When rewards are per-agent dicts (individual_rewards=True), computes
        separate GAE per agent → buffer["advantages"] shape (T, n_agents),
        buffer["returns"] shape (T, n_agents).
        When rewards are scalars (shared), shape is (T,) as before.
        """
        dones = np.array(self.buffer["dones"])

        # Add a final value estimate for bootstrapping
        last_obs = self.buffer["obs"][-1]
        if use_current_device:
            last_value = self._compute_values_on_current_device(last_obs)
        else:
            last_value = self.compute_values(last_obs)

        # Check if rewards are per-agent dicts
        first_reward = self.buffer["rewards"][0]
        individual = isinstance(first_reward, dict)

        if individual:
            # Per-agent GAE (Optie B: shared critic V(s), per-agent rewards)
            agent_ids = sorted(first_reward.keys())
            n_agents = len(agent_ids)
            T = len(self.buffer["rewards"])

            # Build per-agent reward matrix (T, n_agents)
            reward_matrix = np.zeros((T, n_agents), dtype=np.float32)
            for t, r_dict in enumerate(self.buffer["rewards"]):
                for j, aid in enumerate(agent_ids):
                    reward_matrix[t, j] = r_dict.get(aid, 0.0)

            values = np.array(self.buffer["values"])
            values = np.append(values, last_value)

            advantages = np.zeros((T, n_agents), dtype=np.float32)
            returns = np.zeros((T, n_agents), dtype=np.float32)

            # GAE per agent with shared value baseline
            for j in range(n_agents):
                gae = 0.0
                for t in reversed(range(T)):
                    next_val = last_value if t == T - 1 else values[t + 1]
                    delta = reward_matrix[t, j] + self.gamma * next_val * (1 - dones[t]) - values[t]
                    gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                    advantages[t, j] = gae
                    returns[t, j] = advantages[t, j] + values[t]

            self.buffer["advantages"] = advantages
            self.buffer["returns"] = returns
            self.buffer["_individual_rewards"] = True
            self.buffer["_agent_ids"] = agent_ids
        else:
            # Shared reward GAE (original path)
            values = np.array(self.buffer["values"])
            rewards = np.array(self.buffer["rewards"])
            values = np.append(values, last_value)

            advantages = np.zeros_like(rewards, dtype=np.float32)
            returns = np.zeros_like(rewards, dtype=np.float32)

            gae = 0
            for t in reversed(range(len(rewards))):
                next_value = last_value if t == len(rewards) - 1 else values[t + 1]
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
                returns[t] = advantages[t] + values[t]

            self.buffer["advantages"] = advantages
            self.buffer["returns"] = returns
            self.buffer["_individual_rewards"] = False

    def update_policy(self, entropy_coef=0.01):
        """Update policy and value networks using PPO with GPU/MPS acceleration.

        Args:
            entropy_coef: Coefficient for entropy bonus in actor loss. Higher values
                encourage exploration by penalizing deterministic policies.
        """
        # Skip if no data available
        if len(self.buffer["obs"]) == 0:
            return

        print_colored(
            f"[{datetime.now().strftime('%H:%M:%S')}] Updating policy on {self.device}...",
            "yellow",
        )
        update_start_time = time.perf_counter()

        # Move models to device for training
        self._move_models_to_device()

        # Compute advantages and returns (using current device since models are now on GPU/MPS)
        self.compute_advantages_and_returns(use_current_device=True)

        # Get buffer data
        observations = self.buffer["obs"]
        actions = self.buffer["actions"]
        old_action_probs = self.buffer["action_probs"]
        returns = self.buffer["returns"]
        advantages = self.buffer["advantages"]

        # Sub-sample buffer if buffer_size is set (limits gradient steps per episode)
        N_total = len(observations)
        if self.buffer_size is not None and N_total > self.buffer_size:
            sample_idx = np.random.choice(N_total, self.buffer_size, replace=False)
            sample_idx.sort()  # Keep temporal order for LSTM consistency
            observations = [observations[i] for i in sample_idx]
            actions = [actions[i] for i in sample_idx]
            old_action_probs = [old_action_probs[i] for i in sample_idx]
            # returns and advantages may be 2D (T, n_agents) for individual rewards
            returns = returns[sample_idx]
            advantages = advantages[sample_idx]
            self.buffer["rewards"] = [self.buffer["rewards"][i] for i in sample_idx]
            if "gate_log_probs" in self.buffer and len(self.buffer["gate_log_probs"]) == N_total:
                self.buffer["gate_log_probs"] = [self.buffer["gate_log_probs"][i] for i in sample_idx]
            if "gate_actions" in self.buffer and len(self.buffer["gate_actions"]) == N_total:
                self.buffer["gate_actions"] = [self.buffer["gate_actions"][i] for i in sample_idx]
            print_colored(
                f"[{datetime.now().strftime('%H:%M:%S')}] Buffer sub-sampled: {N_total} -> {self.buffer_size} steps",
                "cyan",
            )

        # Normalize advantages (handles both 1D shared and 2D per-agent)
        advantages = np.array(advantages)
        if advantages.ndim == 2:
            # Per-agent: normalize each agent's advantages independently
            for j in range(advantages.shape[1]):
                col = advantages[:, j]
                advantages[:, j] = (col - col.mean()) / (col.std() + 1e-8)
        else:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Prepare data for GPU processing
        print_colored(
            f"[{datetime.now().strftime('%H:%M:%S')}] Preparing data for device transfer...",
            "cyan",
        )

        # Check if we have per-agent advantages (2D) or shared (1D)
        individual_rewards = self.buffer.get("_individual_rewards", False)

        # Convert data to tensors and move to device
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(np.array(returns)).to(self.device)

        # For individual rewards: critic trains on mean returns, actors use per-agent
        if individual_rewards and returns_tensor.ndim == 2:
            critic_returns_tensor = returns_tensor.mean(dim=1)  # (T,) for V(s)
        else:
            critic_returns_tensor = returns_tensor

        # Prepare action and action probability tensors per agent
        agent_ids = [agent.id for agent in self.env.agents]
        N = len(actions)
        n_actions = 2  # volunteer or not
        n_action_features = list(self.actors.values())[0].action_head.out_features

        # --- Extract all actions + probs in a single pass ---
        action_matrix = np.zeros((N, self.n_agents), dtype=np.int64)
        probs_matrix = np.ones((N, self.n_agents, n_action_features), dtype=np.float32)

        for t in range(N):
            act_dict = actions[t]
            prob_dict = old_action_probs[t]
            for idx, aid in enumerate(agent_ids):
                if aid in act_dict:
                    action_matrix[t, idx] = act_dict[aid]
                    prob_val = prob_dict[aid]
                    if isinstance(prob_val, torch.Tensor):
                        probs_matrix[t, idx] = prob_val.numpy()
                    else:
                        probs_matrix[t, idx] = prob_val

        # --- Build action tensor (always needed) ---
        action_tensor_cpu = torch.from_numpy(action_matrix)  # (N, n_agents)

        # --- COMA: Build joint action one-hot tensor via scatter_ ---
        if self.use_coma:
            # Compute scatter indices: agent_idx * n_actions + action
            offsets = torch.arange(self.n_agents).unsqueeze(0) * n_actions  # (1, n_agents)
            scatter_indices = (offsets + action_tensor_cpu).long()  # (N, n_agents)
            joint_action_onehot = torch.zeros(N, self.n_agents * n_actions)
            joint_action_onehot.scatter_(1, scatter_indices, 1.0)
            joint_action_onehot = joint_action_onehot.to(self.device)

        # --- Per-agent action tensors + old prob tensors (sliced from matrices) ---
        action_tensors = {}
        old_action_prob_tensors = {}
        probs_tensor_all = torch.from_numpy(probs_matrix).to(self.device)

        for idx, agent_id in enumerate(agent_ids):
            action_tensors[agent_id] = action_tensor_cpu[:, idx].long().to(self.device)
            old_action_prob_tensors[agent_id] = probs_tensor_all[:, idx]

        # OPTIMIZATION: Pre-tensorize ALL observations before epoch loop
        # This converts numpy dicts -> single GPU tensors ONCE, avoiding per-batch conversion
        print_colored(
            f"[{datetime.now().strftime('%H:%M:%S')}] Pre-tensorizing observations for {self.device}...",
            "cyan",
        )
        pretensorize_start = time.perf_counter()

        # Build observation lists for critic (pre-allocate list of lists)
        preprocessed_rewards_raw = self.buffer["rewards"]
        # For pre-tensorization, rewards must be scalars (used as context feature)
        # If individual rewards (dict), use mean per timestep
        if individual_rewards:
            preprocessed_rewards = [
                float(np.mean(list(r.values()))) if isinstance(r, dict) else r
                for r in preprocessed_rewards_raw
            ]
        else:
            preprocessed_rewards = preprocessed_rewards_raw
        preprocessed_obs_lists = [
            [observations[i][agent.id] for agent in self.env.agents]
            for i in range(N)
        ]

        # Pre-tensorize critic: shape (N, n_agents * obs_dim + 1)
        critic_tensor_all = pretensorize_critic_observations(
            preprocessed_obs_lists, preprocessed_rewards, self.critic.obs_keys
        ).to(self.device)

        if self.use_ic3net:
            # IC3Net pre-tensorization: stack all agents per timestep
            from .communication import pretensorize_ic3net_observations

            # IC3Net also needs scalar rewards for pre-tensorization
            ic3net_tensor_all = pretensorize_ic3net_observations(
                observations, preprocessed_rewards, self.ic3net_actor.obs_keys,
                self.n_agents, agent_ids
            ).to(self.device)

            # Prepare IC3Net action tensors: reuse already-extracted matrices
            ic3net_action_tensor = action_tensor_cpu.long().to(self.device)  # (N, n_agents)
            ic3net_old_action_probs_tensor = probs_tensor_all  # (N, n_agents, n_action_features)

            # Prepare old gate log-probs: (N, n_agents, n_comm_rounds)
            old_gate_log_probs_tensor = torch.stack(
                self.buffer["gate_log_probs"]
            ).to(self.device)

            # Log gate statistics
            gate_actions_all = torch.stack(self.buffer["gate_actions"])  # (N, n_agents, n_rounds)
            gate_open_pct = gate_actions_all.float().mean().item() * 100
            print_colored(
                f"[{datetime.now().strftime('%H:%M:%S')}] IC3Net Gate Stats: {gate_open_pct:.1f}% open",
                "green",
            )
            for agent_idx, agent_id in enumerate(agent_ids):
                agent_gate_pct = gate_actions_all[:, agent_idx, :].float().mean().item() * 100
                print_colored(
                    f"  Agent {agent_id}: {agent_gate_pct:.1f}% open",
                    "green",
                )

            pretensorize_time = time.perf_counter() - pretensorize_start
            print_colored(
                f"[{datetime.now().strftime('%H:%M:%S')}] Pre-tensorization complete in {pretensorize_time:.2f}s "
                f"(critic: {critic_tensor_all.shape}, ic3net: {ic3net_tensor_all.shape})",
                "cyan",
            )
        else:
            # Pre-tensorize actors: shape (N, obs_dim + 1) each
            actor_tensors_all = {}
            agent_masks_all = {}
            for agent_id in self.actors:
                agent_obs_list = []
                agent_rewards_list = []
                mask = []
                for i in range(len(observations)):
                    if agent_id in observations[i]:
                        agent_obs_list.append(observations[i][agent_id])
                        # Extract per-agent reward if dict, else use scalar
                        r = preprocessed_rewards_raw[i]
                        agent_rewards_list.append(
                            r[agent_id] if isinstance(r, dict) else r
                        )
                        mask.append(True)
                    else:
                        # Create dummy observation matching expected shape
                        dummy_obs = {}
                        for key in self.env.observation_space(agent_id).keys():
                            obs_space = self.env.observation_space(agent_id)
                            if key in obs_space and hasattr(obs_space[key], 'sample'):
                                dummy_obs[key] = obs_space[key].sample()
                            else:
                                dummy_obs[key] = 0
                        agent_obs_list.append(dummy_obs)
                        agent_rewards_list.append(0.0)
                        mask.append(False)

                actor_obs_keys = self.actors[agent_id].obs_keys

                actor_tensors_all[agent_id] = pretensorize_actor_observations(
                    agent_obs_list, agent_rewards_list, actor_obs_keys
                ).to(self.device)
                agent_masks_all[agent_id] = torch.tensor(mask, dtype=torch.bool, device=self.device)

            pretensorize_time = time.perf_counter() - pretensorize_start
            print_colored(
                f"[{datetime.now().strftime('%H:%M:%S')}] Pre-tensorization complete in {pretensorize_time:.2f}s "
                f"(critic: {critic_tensor_all.shape}, actors: {next(iter(actor_tensors_all.values())).shape})",
                "cyan",
            )

        print_colored(
            f"[{datetime.now().strftime('%H:%M:%S')}] Starting {self.num_epochs} epochs of training...",
            "cyan",
        )

        # Reset history before batch processing
        self.reset_history()

        # Perform multiple epochs of updates on GPU
        for epoch in range(self.num_epochs):
            epoch_start_time = time.perf_counter()
            print_colored(
                f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch + 1}/{self.num_epochs} - Data size: {len(observations)}",
                "blue",
            )

            # Create random permutation for mini-batching
            indices = torch.randperm(len(observations), device=self.device)

            total_critic_loss = 0.0
            total_actor_loss = 0.0
            total_q_loss = 0.0
            num_batches = 0
            total_batches = (
                len(observations) + self.batch_size - 1
            ) // self.batch_size  # Ceiling division

            print_colored(
                f"[{datetime.now().strftime('%H:%M:%S')}] Starting {total_batches} batches of size {self.batch_size}",
                "blue",
            )

            for start_idx in range(0, len(observations), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(observations))
                batch_indices = indices[start_idx:end_idx]
                num_batches += 1

                # Get batch data via tensor indexing (no Python loops!)
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_critic_returns = critic_returns_tensor[batch_indices]

                # Update critic using pre-tensorized data
                batch_critic_input = critic_tensor_all[batch_indices]
                value_preds = self.critic.forward_batch_pretensorized(batch_critic_input)
                critic_loss = (
                    (value_preds.squeeze() - batch_critic_returns.squeeze()) ** 2
                ).mean()
                total_critic_loss += critic_loss.item()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), max_norm=0.5
                )
                self.critic_optimizer.step()

                # --- COMA: Train Q-network and compute per-agent counterfactual advantages ---
                if self.use_coma:
                    batch_joint_actions = joint_action_onehot[batch_indices]
                    q_preds = self.q_network.forward_batch(batch_critic_input, batch_joint_actions)
                    q_loss = ((q_preds.squeeze() - batch_critic_returns.squeeze()) ** 2).mean()

                    self.q_optimizer.zero_grad()
                    q_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=0.5)
                    self.q_optimizer.step()
                    total_q_loss += q_loss.item()

                    # Compute per-agent counterfactual advantages
                    # For each agent i: A_i = Q(s, a_joint) - b_i
                    # where b_i = sum_a_i [ pi_i(a_i|o_i) * Q(s, (a_{-i}, a_i)) ]
                    with torch.no_grad():
                        q_joint = self.q_network.forward_batch(
                            batch_critic_input, batch_joint_actions
                        ).squeeze()  # (batch,)

                        # Per-agent advantages: (batch, n_agents)
                        per_agent_advantages = torch.zeros(
                            len(batch_indices), self.n_agents, device=self.device
                        )

                        for agent_idx, agent_id in enumerate(agent_ids):
                            # Get current policy probs for this agent
                            if self.use_ic3net:
                                # Will be computed in IC3Net forward pass below
                                pass
                            elif agent_id in self.actors:
                                actor = self.actors[agent_id]
                                batch_actor_input_cf = actor_tensors_all[agent_id][batch_indices]
                                current_probs_cf = actor.forward_batch_pretensorized(batch_actor_input_cf)
                                # current_probs_cf: (batch, n_actions)

                                # Compute counterfactual baseline by marginalizing over agent i's actions
                                baseline_i = torch.zeros(len(batch_indices), device=self.device)
                                for action_val in range(n_actions):
                                    # Create counterfactual joint action: replace agent i's action
                                    cf_actions = batch_joint_actions.clone()
                                    # Zero out agent i's current one-hot
                                    cf_actions[:, agent_idx * n_actions:(agent_idx + 1) * n_actions] = 0
                                    # Set to counterfactual action
                                    cf_actions[:, agent_idx * n_actions + action_val] = 1.0

                                    q_cf = self.q_network.forward_batch(
                                        batch_critic_input, cf_actions
                                    ).squeeze()  # (batch,)
                                    baseline_i += current_probs_cf[:, action_val] * q_cf

                                per_agent_advantages[:, agent_idx] = q_joint - baseline_i

                        # Normalize per-agent advantages
                        for agent_idx in range(self.n_agents):
                            adv = per_agent_advantages[:, agent_idx]
                            per_agent_advantages[:, agent_idx] = (adv - adv.mean()) / (adv.std() + 1e-8)

                if self.use_ic3net:
                    # === IC3Net Actor Update ===
                    batch_ic3net_input = ic3net_tensor_all[batch_indices]
                    batch_ic3net_actions = ic3net_action_tensor[batch_indices]
                    batch_ic3net_old_probs = ic3net_old_action_probs_tensor[batch_indices]
                    batch_old_gate_lp = old_gate_log_probs_tensor[batch_indices]

                    # Forward pass: re-run communication for gradients
                    all_action_probs, all_gate_log_probs = (
                        self.ic3net_actor.forward_batch_pretensorized(batch_ic3net_input)
                    )
                    # all_action_probs: (batch, n_agents, n_actions)
                    # all_gate_log_probs: (batch, n_agents, n_comm_rounds)

                    if self.use_coma:
                        # --- COMA for IC3Net: compute per-agent advantages using IC3Net probs ---
                        with torch.no_grad():
                            for agent_idx in range(self.n_agents):
                                pi_probs = all_action_probs[:, agent_idx, :].detach()
                                baseline_i = torch.zeros(len(batch_indices), device=self.device)
                                for action_val in range(n_actions):
                                    cf_actions = batch_joint_actions.clone()
                                    cf_actions[:, agent_idx * n_actions:(agent_idx + 1) * n_actions] = 0
                                    cf_actions[:, agent_idx * n_actions + action_val] = 1.0
                                    q_cf = self.q_network.forward_batch(
                                        batch_critic_input, cf_actions
                                    ).squeeze()
                                    baseline_i += pi_probs[:, action_val] * q_cf
                                per_agent_advantages[:, agent_idx] = q_joint - baseline_i
                            # Re-normalize
                            for agent_idx in range(self.n_agents):
                                adv = per_agent_advantages[:, agent_idx]
                                per_agent_advantages[:, agent_idx] = (adv - adv.mean()) / (adv.std() + 1e-8)

                    # Compute joint PPO loss across all agents
                    batch_actor_loss = torch.tensor(0.0, device=self.device)

                    for agent_idx in range(self.n_agents):
                        # Use COMA per-agent advantage, per-agent GAE, or shared GAE
                        if self.use_coma:
                            agent_advantage = per_agent_advantages[:, agent_idx]
                        elif individual_rewards and batch_advantages.ndim == 2:
                            agent_advantage = batch_advantages[:, agent_idx]
                        else:
                            agent_advantage = batch_advantages

                        # Current action probs for this agent
                        current_probs = all_action_probs[:, agent_idx, :]
                        agent_actions = batch_ic3net_actions[:, agent_idx]
                        old_probs = batch_ic3net_old_probs[:, agent_idx, :]

                        # Get probability of taken action
                        current_prob_taken = current_probs.gather(
                            1, agent_actions.unsqueeze(1)
                        ).squeeze(1)
                        old_prob_taken = old_probs.gather(
                            1, agent_actions.unsqueeze(1)
                        ).squeeze(1)

                        # Action log-probs
                        new_action_lp = torch.log(current_prob_taken + 1e-8)
                        old_action_lp = torch.log(old_prob_taken + 1e-8)

                        # === PPO ratio: optionally include gate log-probs ===
                        if self.gate_in_ratio:
                            # Coupled ratio: gate + action (stronger gate learning signal)
                            new_gate_lp_sum = all_gate_log_probs[:, agent_idx, :].sum(dim=-1)
                            old_gate_lp_sum = batch_old_gate_lp[:, agent_idx, :].sum(dim=-1)
                            new_joint_lp = new_action_lp + new_gate_lp_sum
                            old_joint_lp = old_action_lp + old_gate_lp_sum
                            action_ratio = torch.exp(new_joint_lp - old_joint_lp)
                        else:
                            # Decoupled ratio: action-only (original behavior)
                            # Gate decisions are NOT included in the importance ratio.
                            # Including gate log-probs doubled the variance of the ratio
                            # and caused training instability / collapse to no-communication.
                            # The gate still learns via gradients flowing through the
                            # communication network (gate → message → LSTM → action_probs).
                            action_ratio = torch.exp(new_action_lp - old_action_lp)

                        # PPO clipped surrogate (action-only ratio)
                        surr1 = action_ratio * agent_advantage
                        surr2 = (
                            torch.clamp(
                                action_ratio,
                                1.0 - self.clip_param,
                                1.0 + self.clip_param,
                            )
                            * agent_advantage
                        )
                        agent_loss = -torch.min(surr1, surr2).mean()

                        # Action entropy bonus
                        entropy = -(current_probs * torch.log(current_probs + 1e-8)).sum(dim=-1).mean()
                        agent_loss = agent_loss - entropy_coef * entropy

                        # Gate entropy bonus: prevent gate collapse (all open or all closed)
                        # Recompute gate probs for this agent from the current forward pass
                        gate_lp = all_gate_log_probs[:, agent_idx, :]  # (batch, n_rounds)
                        gate_p = torch.exp(gate_lp)  # p(gate_taken)
                        gate_p_other = 1.0 - gate_p  # p(other gate action)
                        gate_entropy = -(gate_p * gate_lp + gate_p_other * torch.log(gate_p_other + 1e-8)).mean()
                        agent_loss = agent_loss - self.gate_entropy_coef * entropy_coef * gate_entropy

                        batch_actor_loss = batch_actor_loss + agent_loss

                    total_actor_loss += batch_actor_loss.item()

                    # Single optimizer step for shared IC3Net
                    self.ic3net_optimizer.zero_grad()
                    batch_actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.ic3net_actor.parameters(), max_norm=0.5
                    )
                    self.ic3net_optimizer.step()

                else:
                    # === Standard MAPPO Actor Updates ===
                    agent_id_to_idx = {aid: idx for idx, aid in enumerate(agent_ids)}
                    for agent_id in self.actors:
                        actor = self.actors[agent_id]
                        optimizer = self.actor_optimizers[agent_id]
                        agent_idx = agent_id_to_idx[agent_id]

                        # Get batch data for this agent via tensor indexing
                        batch_agent_actions = action_tensors[agent_id][batch_indices]
                        batch_old_action_probs = old_action_prob_tensors[agent_id][batch_indices]
                        batch_actor_input = actor_tensors_all[agent_id][batch_indices]
                        batch_mask = agent_masks_all[agent_id][batch_indices]

                        # Forward pass using pre-tensorized data
                        current_action_probs = actor.forward_batch_pretensorized(batch_actor_input)

                        if batch_mask.sum() > 0:
                            valid_indices = batch_mask.nonzero(as_tuple=True)[0]
                            valid_current_probs = current_action_probs[valid_indices]
                            valid_actions = batch_agent_actions[valid_indices]
                            valid_old_probs = batch_old_action_probs[valid_indices]
                            # Use COMA per-agent advantages, per-agent GAE, or shared GAE
                            if self.use_coma:
                                valid_advantages = per_agent_advantages[valid_indices, agent_idx]
                            elif individual_rewards and batch_advantages.ndim == 2:
                                valid_advantages = batch_advantages[valid_indices, agent_idx]
                            else:
                                valid_advantages = batch_advantages[valid_indices]

                            # Get probabilities for taken actions
                            current_action_prob_taken = valid_current_probs.gather(
                                1, valid_actions.unsqueeze(1)
                            ).squeeze(1)
                            old_action_prob_taken = valid_old_probs.gather(
                                1, valid_actions.unsqueeze(1)
                            ).squeeze(1)

                            # Compute ratio
                            ratio = current_action_prob_taken / (
                                old_action_prob_taken + 1e-8
                            )

                            # Compute surrogate losses
                            surrogate1 = ratio * valid_advantages
                            surrogate2 = (
                                torch.clamp(
                                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                                )
                                * valid_advantages
                            )

                            # Actor loss (PPO clipped surrogate)
                            actor_loss = -torch.min(surrogate1, surrogate2).mean()

                            # Entropy bonus: encourages exploration by penalizing
                            # deterministic policies. H(π) = -Σ p(a) log p(a)
                            entropy = -(valid_current_probs * torch.log(valid_current_probs + 1e-8)).sum(dim=-1).mean()
                            actor_loss = actor_loss - entropy_coef * entropy

                            total_actor_loss += actor_loss.item()

                            # Update actor
                            optimizer.zero_grad()
                            actor_loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                actor.parameters(), max_norm=0.5
                            )
                            optimizer.step()

                # Print progress
                if (
                    num_batches % max(1, total_batches // 3) == 0
                    or num_batches == total_batches
                ):
                    progress_pct = (num_batches / total_batches) * 100
                    elapsed_time = time.perf_counter() - epoch_start_time
                    avg_critic_loss_so_far = (
                        total_critic_loss / num_batches if num_batches > 0 else 0
                    )
                    avg_actor_loss_so_far = (
                        total_actor_loss / num_batches if num_batches > 0 else 0
                    )
                    avg_q_loss_so_far = total_q_loss / num_batches if num_batches > 0 else 0
                    print_colored(
                        f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch + 1} - "
                        f"Batch {num_batches}/{total_batches} ({progress_pct:.1f}%) - "
                        f"Time: {elapsed_time:.1f}s - Critic: {avg_critic_loss_so_far:.6f} - Q: {avg_q_loss_so_far:.6f}"
                        + (f" - Actor: {avg_actor_loss_so_far:.6f}" if self.use_ic3net else ""),
                        "purple",
                    )

            epoch_time = time.perf_counter() - epoch_start_time
            avg_critic_loss = total_critic_loss / num_batches if num_batches > 0 else 0
            avg_q_loss = total_q_loss / num_batches if num_batches > 0 else 0
            print_colored(
                f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch + 1} complete - "
                f"Time: {epoch_time:.2f}s, Critic: {avg_critic_loss:.6f}, Q: {avg_q_loss:.6f}",
                "green",
            )

        total_update_time = time.perf_counter() - update_start_time
        print_colored(
            f"[{datetime.now().strftime('%H:%M:%S')}] Policy update complete - Total time: {total_update_time:.2f}s",
            "yellow",
        )

        # Move models back to CPU for next simulation episodes
        self._move_models_to_cpu()

        # Clear the buffer after updating
        self._clear_buffer()

    def _clear_buffer(self):
        """Clear the experience buffer."""
        self.buffer = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "values": [],
            "action_probs": [],
            "advantages": [],
            "returns": [],
        }
        if self.use_ic3net:
            self.buffer["gate_log_probs"] = []
            self.buffer["gate_actions"] = []

    def save_models(self, path, save_optimizers=True):
        """Save model weights and optionally optimizer states to the specified path."""
        # Save critic
        # Check if directory exists, if not create it
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.critic.state_dict(), f"{path}/critic.pt")
        if self.q_network is not None:
            torch.save(self.q_network.state_dict(), f"{path}/q_network.pt")

        # Save actors
        for agent_id, actor in self.actors.items():
            torch.save(actor.state_dict(), f"{path}/actor_{agent_id}.pt")

        # Save IC3Net actor if enabled
        if self.use_ic3net:
            torch.save(self.ic3net_actor.state_dict(), f"{path}/ic3net_actor.pt")

        # Save optimizer states (needed for resume)
        if save_optimizers:
            torch.save(self.critic_optimizer.state_dict(), f"{path}/critic_optimizer.pt")
            if self.use_ic3net:
                torch.save(self.ic3net_optimizer.state_dict(), f"{path}/ic3net_optimizer.pt")
            else:
                for agent_id, opt in self.actor_optimizers.items():
                    torch.save(opt.state_dict(), f"{path}/actor_optimizer_{agent_id}.pt")
            if self.q_optimizer is not None:
                torch.save(self.q_optimizer.state_dict(), f"{path}/q_optimizer.pt")

        # Save device info
        with open(f"{path}/device.txt", "w") as f:
            f.write(str(self.device))

    def load_models(self, path, load_optimizers=False):
        """Load model weights and optionally optimizer states from the specified path."""
        # Load critic
        self.critic.load_state_dict(
            torch.load(f"{path}/critic.pt", map_location=self.device)
        )
        # Load Q-network if available and COMA enabled
        if self.q_network is not None:
            q_path = f"{path}/q_network.pt"
            if os.path.exists(q_path):
                self.q_network.load_state_dict(
                    torch.load(q_path, map_location=self.device)
                )

        # Load actors
        for agent_id, actor in self.actors.items():
            actor.load_state_dict(
                torch.load(f"{path}/actor_{agent_id}.pt", map_location=self.device)
            )

        # Load IC3Net actor if enabled
        if self.use_ic3net:
            ic3net_path = f"{path}/ic3net_actor.pt"
            if os.path.exists(ic3net_path):
                self.ic3net_actor.load_state_dict(
                    torch.load(ic3net_path, map_location=self.device)
                )

        # Load optimizer states (for resume)
        if load_optimizers:
            crit_opt_path = f"{path}/critic_optimizer.pt"
            if os.path.exists(crit_opt_path):
                self.critic_optimizer.load_state_dict(
                    torch.load(crit_opt_path, map_location=self.device)
                )
            if self.use_ic3net:
                ic3_opt_path = f"{path}/ic3net_optimizer.pt"
                if os.path.exists(ic3_opt_path):
                    self.ic3net_optimizer.load_state_dict(
                        torch.load(ic3_opt_path, map_location=self.device)
                    )
            else:
                for agent_id, opt in self.actor_optimizers.items():
                    opt_path = f"{path}/actor_optimizer_{agent_id}.pt"
                    if os.path.exists(opt_path):
                        opt.load_state_dict(
                            torch.load(opt_path, map_location=self.device)
                        )
            if self.q_optimizer is not None:
                q_opt_path = f"{path}/q_optimizer.pt"
                if os.path.exists(q_opt_path):
                    self.q_optimizer.load_state_dict(
                        torch.load(q_opt_path, map_location=self.device)
                    )

        # After loading, move models to CPU for simulation
        self._move_models_to_cpu()

    def _move_models_to_cpu(self):
        """Move all models to CPU for inference during simulation.

        Also sets models to eval mode to disable dropout during inference.
        """
        if not hasattr(self, "_models_on_cpu") or not self._models_on_cpu:
            print_colored(
                f"[{datetime.now().strftime('%H:%M:%S')}] Moving models to CPU for inference (eval mode)...",
                "cyan",
            )
            start_time = time.perf_counter()
            for actor in self.actors.values():
                actor.to(torch.device("cpu"))
                actor.eval()
            self.critic.to(torch.device("cpu"))
            self.critic.eval()
            if self.q_network is not None:
                self.q_network.to(torch.device("cpu"))
                self.q_network.eval()
            if self.use_ic3net:
                self.ic3net_actor.to(torch.device("cpu"))
                self.ic3net_actor.eval()
            transfer_time = time.perf_counter() - start_time
            print_colored(
                f"[{datetime.now().strftime('%H:%M:%S')}] Models moved to CPU (eval mode) in {transfer_time:.3f}s",
                "cyan",
            )
            self._models_on_cpu = True

    def _move_models_to_device(self):
        """Move all models to GPU/MPS device for training.

        Also sets models to train mode to enable dropout during training.
        """
        if not hasattr(self, "_models_on_cpu") or self._models_on_cpu:
            print_colored(
                f"[{datetime.now().strftime('%H:%M:%S')}] Moving models to {self.device} for training (train mode)...",
                "cyan",
            )
            start_time = time.perf_counter()
            for actor in self.actors.values():
                actor.to(self.device)
                actor.train()
            self.critic.to(self.device)
            self.critic.train()
            if self.q_network is not None:
                self.q_network.to(self.device)
                self.q_network.train()
            if self.use_ic3net:
                self.ic3net_actor.to(self.device)
                self.ic3net_actor.train()
            transfer_time = time.perf_counter() - start_time
            print_colored(
                f"[{datetime.now().strftime('%H:%M:%S')}] Models moved to {self.device} (train mode) in {transfer_time:.3f}s",
                "cyan",
            )
            self._models_on_cpu = False
            self._ensure_optimizers_on_device()

    def _ensure_optimizers_on_device(self):
        """Ensure optimizer states (momentum buffers etc.) are on the correct device.

        PyTorch optimizers do NOT automatically move their internal state tensors
        when model.to(device) is called. This causes 'tensors on different devices'
        errors when resuming a checkpoint trained on a different device (e.g. CPU → MPS).
        """
        def _move_optimizer_state(optimizer, device):
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        _move_optimizer_state(self.critic_optimizer, self.device)

        if self.use_ic3net and hasattr(self, 'ic3net_optimizer'):
            _move_optimizer_state(self.ic3net_optimizer, self.device)
        else:
            for opt in self.actor_optimizers.values():
                _move_optimizer_state(opt, self.device)

        if self.q_optimizer is not None:
            _move_optimizer_state(self.q_optimizer, self.device)
