from datetime import datetime
import time
import torch
import torch.optim as optim
import numpy as np
from collections import deque
import os

from ...core.display import print_colored
from ...core.hyperparameters import DROPOUT_RATE, WEIGHT_INIT
from ...utils.gpu_utils import (
    get_optimal_batch_size,
    setup_mixed_precision,
    print_gpu_utilization,
    optimize_for_h200,
    estimate_model_size,
)

from .networks import ActorNetwork, CriticNetwork


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
        num_epochs=10,
        dropout_rate=None,
        weight_init=None,
        device=None,
        use_mixed_precision=True,
        auto_batch_size=True,
    ):
        self.dropout_rate = dropout_rate if dropout_rate is not None else DROPOUT_RATE
        self.weight_init = weight_init if weight_init is not None else WEIGHT_INIT
        self.env = env
        self.n_agents = len(env.agents)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.base_batch_size = batch_size  # Store original for auto-sizing
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.auto_batch_size = auto_batch_size

        # Set device for GPU/MPS acceleration
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        print_colored(f"MAPPOAgent using device: {self.device}", "blue")

        # Apply H200-specific optimizations
        if self.device.type == "cuda":
            opt_result = optimize_for_h200(self.device)
            if opt_result.get("optimized"):
                print_colored(f"✓ H200 optimizations enabled (TF32: {opt_result.get('tf32_enabled')})", "green")

        # Setup mixed precision training
        self.scaler, self.amp_dtype = setup_mixed_precision(self.device, use_mixed_precision)
        self.use_mixed_precision = (self.scaler is not None or self.amp_dtype != torch.float32)

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

        # Create centralized critic
        first_agent = env.agents[0]
        self.critic = CriticNetwork(
            env.observation_space(first_agent.id),
            self.n_agents,
            hidden_size=2 * hidden_size,
            dropout_rate=self.dropout_rate,
            weight_init=self.weight_init,
            device=self.device,
        ).to(self.device)

        # Setup optimizers
        self.actor_optimizers = {
            agent_id: optim.Adam(actor.parameters(), lr=lr_actor)
            for agent_id, actor in self.actors.items()
        }
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Estimate total model size for batch sizing
        total_model_size = sum(estimate_model_size(actor) for actor in self.actors.values())
        total_model_size += estimate_model_size(self.critic)
        self.model_size_mb = total_model_size
        print_colored(f"Total model size: {total_model_size:.2f} MB", "blue")

        # Hybrid approach: CPU for inference, GPU for training
        self.inference_device = torch.device("cpu")  # Always use CPU for fast single-observation inference
        self.training_device = self.device  # Use specified device for training

        if self.training_device.type == "cuda":
            print_colored(f"Hybrid mode: CPU inference (fast single obs) + GPU training (fast batches)", "green")
            # Start with models on CPU for inference
            for actor in self.actors.values():
                actor.to(self.inference_device)
            self.critic.to(self.inference_device)
        else:
            # For CPU/MPS, no need for hybrid
            print_colored(f"Models will remain on {self.device} for both inference and training", "green")
            self.inference_device = self.device

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

        # For tracking training performance
        self.episode_rewards = deque(maxlen=100)

    def select_actions(self, observations, deterministic=False):
        """Select actions for all agents based on their observations."""
        actions = {}
        action_probs = {}

        # Models are already on inference_device (CPU for CUDA training)
        with torch.no_grad():
            for agent_id, obs in observations.items():
                if agent_id in self.actors:
                    # Get the reward for this agent from the last step
                    reward = (
                        self.buffer["rewards"][-1] if self.buffer["rewards"] else None
                    )
                    action, probs = self.actors[agent_id].act(
                        obs, reward, deterministic
                    )
                    actions[agent_id] = action  # action is already int
                    action_probs[agent_id] = probs  # Already on CPU

        return actions, action_probs

    def compute_values(self, observations):
        """Compute values for the current observations using the critic network."""
        # Convert observations dict to list in agent order
        obs_list = [observations[agent.id] for agent in self.env.agents]
        # Get the reward from the last step
        reward = self.buffer["rewards"][-1] if self.buffer["rewards"] else None

        # Critic is already on inference_device (CPU for CUDA training)
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

    def reset_history(self):
        """Reset the history buffers for all networks."""
        for actor in self.actors.values():
            actor.reset_history()
        self.critic.reset_history()

    def compute_advantages_and_returns(self):
        """Compute GAE advantages and returns for stored trajectories."""
        values = np.array(self.buffer["values"])
        rewards = np.array(self.buffer["rewards"])
        dones = np.array(self.buffer["dones"])

        # Add a final value estimate for bootstrapping
        last_obs = self.buffer["obs"][-1]
        last_value = self.compute_values(last_obs)
        values = np.append(values, last_value)

        # Initialize advantages and returns
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)

        # Compute GAE advantages and returns
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        self.buffer["advantages"] = advantages  # type: ignore
        self.buffer["returns"] = returns  # type: ignore

    def update_policy(self):
        """Update policy and value networks using PPO with GPU/MPS acceleration."""
        # Skip if no data available
        if len(self.buffer["obs"]) == 0:
            return

        update_start_time = time.perf_counter()

        # Move models to training device (GPU) if using hybrid mode
        if self.training_device != self.inference_device:
            transfer_start = time.perf_counter()
            for actor in self.actors.values():
                actor.to(self.training_device)
            self.critic.to(self.training_device)
            transfer_time = time.perf_counter() - transfer_start
            print_colored(f"  Transferred models to GPU in {transfer_time:.2f}s", "cyan")

        # Compute advantages and returns (now on training device)
        self.compute_advantages_and_returns()

        num_samples = len(self.buffer["obs"])

        # Auto-adjust batch size based on episode length and GPU memory
        if self.auto_batch_size:
            self.batch_size = get_optimal_batch_size(
                num_samples=num_samples,
                device=self.device,
                model_size_mb=self.model_size_mb,
            )
            print_colored(
                f"Auto-adjusted batch size: {self.batch_size} (for {num_samples} samples)",
                "cyan",
            )

        # Get buffer data
        observations = self.buffer["obs"]
        actions = self.buffer["actions"]
        old_action_probs = self.buffer["action_probs"]
        returns = self.buffer["returns"]
        advantages = self.buffer["advantages"]

        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert data to tensors and move to training device
        advantages_tensor = torch.FloatTensor(advantages).to(self.training_device)
        returns_tensor = torch.FloatTensor(returns).to(self.training_device)

        # Prepare action and action probability tensors
        action_tensors = {}
        old_action_prob_tensors = {}

        for agent_id in self.actors:
            agent_actions = []
            agent_old_probs = []

            for i in range(len(actions)):
                if agent_id in actions[i]:
                    agent_actions.append(actions[i][agent_id])
                    agent_old_probs.append(old_action_probs[i][agent_id])
                else:
                    agent_actions.append(0)  # Default action
                    agent_old_probs.append(
                        torch.ones(self.actors[agent_id].action_head.out_features)
                    )

            action_tensors[agent_id] = torch.LongTensor(agent_actions).to(self.training_device)
            # Stack old action probabilities and move to training device
            old_action_prob_tensors[agent_id] = torch.stack(agent_old_probs).to(
                self.training_device
            )

        # Reset history before batch processing
        self.reset_history()

        # Perform multiple epochs of updates on GPU
        print_colored(f"  Training for {self.num_epochs} epochs:", "cyan")

        for epoch in range(self.num_epochs):
            epoch_start_time = time.perf_counter()

            # Create random permutation for mini-batching
            indices = torch.randperm(len(observations), device=self.training_device)

            total_critic_loss = 0.0
            total_actor_losses = {agent_id: 0.0 for agent_id in self.actors}
            num_batches = 0
            total_batches = (
                len(observations) + self.batch_size - 1
            ) // self.batch_size  # Ceiling division

            for start_idx in range(0, len(observations), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(observations))
                batch_indices = indices[start_idx:end_idx]
                num_batches += 1

                # Convert batch indices to CPU for data access
                batch_indices_cpu = batch_indices.cpu().numpy()

                # Get batch data
                batch_obs = [observations[i] for i in batch_indices_cpu]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # Update critic (centralized) - optimized batch processing
                if len(batch_obs) > 0:
                    # Prepare batch data for critic
                    batch_obs_lists = []
                    batch_rewards = []

                    for j, obs_dict in enumerate(batch_obs):
                        obs_list = [obs_dict[agent.id] for agent in self.env.agents]
                        batch_idx = batch_indices_cpu[j]
                        reward = self.buffer["rewards"][batch_idx]
                        batch_obs_lists.append(obs_list)
                        batch_rewards.append(reward)

                    self.critic_optimizer.zero_grad()

                    # Use mixed precision if enabled
                    if self.use_mixed_precision and self.device.type == "cuda":
                        with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                            value_preds = self.critic.forward_batch(
                                batch_obs_lists, batch_rewards
                            )
                            critic_loss = (
                                (value_preds.squeeze() - batch_returns.squeeze()) ** 2
                            ).mean()

                        # Scale loss for FP16 if using scaler
                        if self.scaler is not None:
                            self.scaler.scale(critic_loss).backward()
                            self.scaler.unscale_(self.critic_optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.critic.parameters(), max_norm=0.5
                            )
                            self.scaler.step(self.critic_optimizer)
                            self.scaler.update()
                        else:
                            # BF16 doesn't need scaling
                            critic_loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                self.critic.parameters(), max_norm=0.5
                            )
                            self.critic_optimizer.step()
                    else:
                        # Standard FP32 training
                        value_preds = self.critic.forward_batch(
                            batch_obs_lists, batch_rewards
                        )
                        critic_loss = (
                            (value_preds.squeeze() - batch_returns.squeeze()) ** 2
                        ).mean()
                        critic_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.critic.parameters(), max_norm=0.5
                        )
                        self.critic_optimizer.step()

                    total_critic_loss += critic_loss.item()

                # Update actors (decentralized) - optimized batch processing
                for agent_id in self.actors:
                    actor = self.actors[agent_id]
                    optimizer = self.actor_optimizers[agent_id]

                    # Get batch data for this agent
                    batch_agent_actions = action_tensors[agent_id][batch_indices]
                    batch_old_action_probs = old_action_prob_tensors[agent_id][
                        batch_indices
                    ]

                    # Prepare batch data for this actor
                    batch_agent_obs = []
                    batch_agent_rewards = []
                    agent_mask = []  # Track which observations have this agent

                    for j, obs_dict in enumerate(batch_obs):
                        if agent_id in obs_dict:
                            batch_agent_obs.append(obs_dict[agent_id])
                            batch_idx = batch_indices_cpu[j]
                            batch_agent_rewards.append(
                                self.buffer["rewards"][batch_idx]
                            )
                            agent_mask.append(True)
                        else:
                            # Use dummy observation for missing agents
                            dummy_obs = {}
                            for key in self.env.observation_space(agent_id).keys():
                                if hasattr(
                                    self.env.observation_space(agent_id)[key], "sample"
                                ):
                                    dummy_obs[key] = self.env.observation_space(
                                        agent_id
                                    )[key].sample()
                                else:
                                    dummy_obs[key] = 0
                            batch_agent_obs.append(dummy_obs)
                            batch_agent_rewards.append(0.0)
                            agent_mask.append(False)

                    # Use batch forward pass for actor - single network call for entire batch
                    if batch_agent_obs:
                        # Filter out dummy entries first
                        agent_mask_tensor = torch.tensor(agent_mask, device=self.training_device)
                        if agent_mask_tensor.sum() > 0:
                            optimizer.zero_grad()

                            # Use mixed precision if enabled
                            if self.use_mixed_precision and self.device.type == "cuda":
                                with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                                    current_action_probs = actor.forward_batch(
                                        batch_agent_obs, batch_agent_rewards
                                    )

                                    # Only process entries where agent was actually present
                                    valid_indices = agent_mask_tensor.nonzero(as_tuple=True)[0]
                                    valid_current_probs = current_action_probs[valid_indices]
                                    valid_actions = batch_agent_actions[valid_indices]
                                    valid_old_probs = batch_old_action_probs[valid_indices]
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

                                    # Actor loss
                                    actor_loss = -torch.min(surrogate1, surrogate2).mean()

                                # Scale and backward
                                if self.scaler is not None:
                                    self.scaler.scale(actor_loss).backward()
                                    self.scaler.unscale_(optimizer)
                                    torch.nn.utils.clip_grad_norm_(
                                        actor.parameters(), max_norm=0.5
                                    )
                                    self.scaler.step(optimizer)
                                    self.scaler.update()
                                else:
                                    actor_loss.backward()
                                    torch.nn.utils.clip_grad_norm_(
                                        actor.parameters(), max_norm=0.5
                                    )
                                    optimizer.step()
                            else:
                                # Standard FP32 training
                                current_action_probs = actor.forward_batch(
                                    batch_agent_obs, batch_agent_rewards
                                )

                                valid_indices = agent_mask_tensor.nonzero(as_tuple=True)[0]
                                valid_current_probs = current_action_probs[valid_indices]
                                valid_actions = batch_agent_actions[valid_indices]
                                valid_old_probs = batch_old_action_probs[valid_indices]
                                valid_advantages = batch_advantages[valid_indices]

                                current_action_prob_taken = valid_current_probs.gather(
                                    1, valid_actions.unsqueeze(1)
                                ).squeeze(1)
                                old_action_prob_taken = valid_old_probs.gather(
                                    1, valid_actions.unsqueeze(1)
                                ).squeeze(1)

                                ratio = current_action_prob_taken / (
                                    old_action_prob_taken + 1e-8
                                )

                                surrogate1 = ratio * valid_advantages
                                surrogate2 = (
                                    torch.clamp(
                                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                                    )
                                    * valid_advantages
                                )

                                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                                actor_loss.backward()
                                torch.nn.utils.clip_grad_norm_(
                                    actor.parameters(), max_norm=0.5
                                )
                                optimizer.step()

                            total_actor_losses[agent_id] += actor_loss.item()

            epoch_time = time.perf_counter() - epoch_start_time
            avg_critic_loss = total_critic_loss / num_batches if num_batches > 0 else 0
            print_colored(
                f"    Epoch {epoch + 1}/{self.num_epochs}: Loss={avg_critic_loss:.2e}, Time={epoch_time:.1f}s",
                "white",
            )

        # Move models back to inference device (CPU) if using hybrid mode
        if self.training_device != self.inference_device:
            transfer_start = time.perf_counter()
            for actor in self.actors.values():
                actor.to(self.inference_device)
            self.critic.to(self.inference_device)
            transfer_time = time.perf_counter() - transfer_start
            print_colored(f"  Transferred models back to CPU in {transfer_time:.2f}s", "cyan")

        # Clear the buffer after updating
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

    def save_models(self, path):
        """Save model weights to the specified path."""
        # Save critic
        # Check if directory exists, if not create it
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.critic.state_dict(), f"{path}/critic.pt")

        # Save actors
        for agent_id, actor in self.actors.items():
            torch.save(actor.state_dict(), f"{path}/actor_{agent_id}.pt")

        # Save device info
        with open(f"{path}/device.txt", "w") as f:
            f.write(str(self.device))

    def load_models(self, path):
        """Load model weights from the specified path."""
        # Load to inference device (CPU for hybrid mode)
        load_device = self.inference_device if hasattr(self, 'inference_device') else self.device

        # Load critic
        self.critic.load_state_dict(
            torch.load(f"{path}/critic.pt", map_location=load_device)
        )

        # Load actors
        for agent_id, actor in self.actors.items():
            actor.load_state_dict(
                torch.load(f"{path}/actor_{agent_id}.pt", map_location=load_device)
            )

        # Ensure models are on inference device
        self.critic.to(load_device)
        for actor in self.actors.values():
            actor.to(load_device)

        print_colored(f"Models loaded and ready on {load_device}", "green")
