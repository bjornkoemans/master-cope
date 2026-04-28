"""Communication Modules for MAPPO.

Contains two communication-capable actor architectures:
1. IC3NetActor — LSTM-based (Singh, Jain & Sukhbaatar, ICLR 2019)
2. MLPCommActor — MLP-based (same comm mechanism, no temporal memory)

Both support comm_disabled mode for fair no-communication baselines.

Key features:
- Hard binary gating: each agent learns when to communicate
- Mean-pooled message aggregation (excluding self)
- Parameter sharing across all agents
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces


def pretensorize_ic3net_observations(
    obs_dicts_list: list[dict],
    rewards: list[float],
    obs_keys: list[str],
    n_agents: int,
    agent_ids: list[int],
) -> torch.Tensor:
    """Convert per-timestep observation dicts to a stacked tensor for all agents.

    Args:
        obs_dicts_list: List of observation dicts per timestep,
                        each {agent_id: obs_dict}
        rewards: List of scalar rewards per timestep
        obs_keys: Ordered list of observation key names
        n_agents: Number of agents
        agent_ids: Ordered list of agent IDs

    Returns:
        Tensor of shape (N, n_agents, obs_dim + 1) on CPU, dtype float32
        where obs_dim + 1 includes the reward as the last feature
    """
    n_timesteps = len(obs_dicts_list)

    # Compute feature layout from first sample
    first_obs = obs_dicts_list[0][agent_ids[0]]
    feature_layout = []  # list of (key, size)
    obs_dim = 0
    for key in obs_keys:
        if key in first_obs:
            val = first_obs[key]
            size = val.size if isinstance(val, np.ndarray) else 1
            feature_layout.append((key, size))
            obs_dim += size
    total_dim = obs_dim + 1  # +1 for reward

    # Pre-allocate and fill in-place
    result = np.empty((n_timesteps, n_agents, total_dim), dtype=np.float32)

    for t in range(n_timesteps):
        obs_dict = obs_dicts_list[t]
        reward = rewards[t] if rewards is not None else 0.0

        for agent_idx, agent_id in enumerate(agent_ids):
            if agent_id in obs_dict:
                agent_obs = obs_dict[agent_id]
                col = 0
                for key, size in feature_layout:
                    val = agent_obs[key]
                    if size > 1:
                        result[t, agent_idx, col:col + size] = val.flat
                    else:
                        result[t, agent_idx, col] = float(val) if isinstance(val, np.ndarray) else val
                    col += size
            result[t, agent_idx, obs_dim] = reward

    return torch.from_numpy(result)


class IC3NetActor(nn.Module):
    """IC3Net communication-enabled actor network with parameter sharing.

    All agents share this single network. They are distinguished by:
    - Their individual observations
    - Their persistent LSTM hidden/cell states
    - Their gate decisions

    Architecture follows the official IC3Net implementation (comm.py):
    - encoder: obs → hidden_size
    - gate_head: hidden → 2-class gate logits (hard binary gate)
    - C_module: hidden → hidden (communication transform)
    - f_module: LSTMCell(hidden_size, hidden_size) with additive comm input
    - action_head: hidden → action logits
    """

    def __init__(
        self,
        obs_space,
        action_space,
        n_agents: int,
        hidden_size: int = 64,
        n_comm_rounds: int = 2,
        comm_dropout: float = 0.0,
        weight_init: str = "xavier_uniform",
        comm_disabled: bool = False,
    ):
        super(IC3NetActor, self).__init__()
        self.n_agents = n_agents
        self.hidden_size = hidden_size
        self.n_comm_rounds = n_comm_rounds
        self.comm_disabled = comm_disabled

        # Communication warmup: disable comm for the first N episodes
        # so the base policy can learn without noisy messages.
        # Set via set_comm_warmup() after construction.
        self._comm_warmup_episodes: int = 0
        self._current_episode: int = 0

        # Calculate observation input size
        obs_input_dim = 0
        self.obs_keys = []
        for key, space in obs_space.items():
            self.obs_keys.append(key)
            if isinstance(space, spaces.Discrete):
                obs_input_dim += 1
            elif isinstance(space, spaces.Box):
                obs_input_dim += int(np.prod(space.shape))
            elif isinstance(space, spaces.MultiBinary):
                obs_input_dim += int(np.prod(space.shape))

        # +1 for reward
        self.obs_input_dim = obs_input_dim + 1

        # === Network modules ===
        # Encoder: obs+reward → hidden
        self.encoder = nn.Linear(self.obs_input_dim, hidden_size)

        # Gate head: hidden → 2 logits (communicate / don't communicate)
        self.gate_head = nn.Linear(hidden_size, 2)

        # Communication transform (shared across rounds, following official code)
        self.C_module = nn.Linear(hidden_size, hidden_size)

        # LSTM cell (shared across rounds, following official code)
        # Input is hidden_size because we use additive combination (x + comm)
        self.f_module = nn.LSTMCell(hidden_size, hidden_size)

        # Action head
        self.action_head = nn.Linear(hidden_size, action_space.n)

        # Communication dropout
        self.comm_dropout = nn.Dropout(p=comm_dropout)

        # Communication mask: prevents self-communication
        # Shape: (n_agents, n_agents) — 1 everywhere except diagonal
        self.register_buffer(
            "comm_mask",
            torch.ones(n_agents, n_agents) - torch.eye(n_agents),
        )

        # Persistent LSTM states (for rollout, not training)
        self._hidden_states: torch.Tensor | None = None  # (n_agents, hidden_size)
        self._cell_states: torch.Tensor | None = None    # (n_agents, hidden_size)

        # Store last gate info for logging/buffer (set during forward/act)
        self._last_gate_log_probs: torch.Tensor | None = None  # (n_agents, n_comm_rounds)
        self._last_gate_actions: torch.Tensor | None = None     # (n_agents, n_comm_rounds)

        # Initialize weights
        self._init_weights(weight_init)

    def _init_weights(self, method: str = "xavier_uniform"):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if method == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif method == "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                elif method == "orthogonal":
                    nn.init.orthogonal_(module.weight)
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # LSTMCell has special initialization
        for name, param in self.f_module.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
                # Set forget gate bias to 1 (helps LSTM remember)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

    def set_comm_warmup(self, warmup_episodes: int) -> None:
        """Set how many episodes to disable comm at the start of training."""
        self._comm_warmup_episodes = warmup_episodes

    def set_current_episode(self, episode: int) -> None:
        """Update the current episode counter (for comm warmup)."""
        self._current_episode = episode

    @property
    def _comm_active(self) -> bool:
        """Whether communication is currently active (past warmup and not disabled)."""
        if self.comm_disabled:
            return False
        if self._current_episode < self._comm_warmup_episodes:
            return False
        return True

    def reset_hidden_states(self):
        """Reset LSTM hidden and cell states. Call at the start of each episode."""
        self._hidden_states = None
        self._cell_states = None

    def _encode_observations(self, all_obs_dicts: dict, reward) -> torch.Tensor:
        """Encode observations for all agents into feature vectors.

        Args:
            all_obs_dicts: Dict {agent_id: obs_dict}
            reward: Current reward — float (shared) or dict[int, float] (per-agent)

        Returns:
            Tensor of shape (n_agents, obs_input_dim) on model's device
        """
        current_device = next(self.parameters()).device
        agent_features = []

        for agent_id in sorted(all_obs_dicts.keys()):
            obs_dict = all_obs_dicts[agent_id]
            parts = []
            for key in self.obs_keys:
                if key in obs_dict:
                    val = obs_dict[key]
                    if isinstance(val, np.ndarray):
                        parts.append(
                            torch.from_numpy(val.flatten().astype(np.float32))
                        )
                    else:
                        parts.append(torch.tensor([val], dtype=torch.float32))

            # Append reward (per-agent if dict, shared if float)
            if isinstance(reward, dict):
                r = reward.get(agent_id, 0.0)
            else:
                r = reward if reward is not None else 0.0
            parts.append(torch.tensor([r], dtype=torch.float32))
            agent_features.append(torch.cat(parts).to(current_device))

        return torch.stack(agent_features)  # (n_agents, obs_input_dim)

    def forward(
        self,
        all_obs_dicts: dict,
        reward: float | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with IC3Net communication for all agents.

        Used during rollout (CPU inference). Maintains persistent LSTM states.

        Args:
            all_obs_dicts: Dict {agent_id: obs_dict} for all agents
            reward: Current step reward
            deterministic: If True, use argmax for gate instead of sampling

        Returns:
            action_probs: (n_agents, n_actions) — action probability distributions
            gate_log_probs: (n_agents, n_comm_rounds) — log-probs of gate decisions
            gate_actions: (n_agents, n_comm_rounds) — binary gate decisions
        """
        current_device = next(self.parameters()).device
        n = self.n_agents

        # 1. Encode observations
        obs_flat = self._encode_observations(all_obs_dicts, reward)  # (n, obs_dim)
        x = F.relu(self.encoder(obs_flat))  # (n, hidden_size)

        # 2. Initialize or use persistent hidden states
        if self._hidden_states is None:
            h = x.clone()
            c = torch.zeros(n, self.hidden_size, device=current_device)
        else:
            h = self._hidden_states.to(current_device)
            c = self._cell_states.to(current_device)

        # 3. Communication rounds
        all_gate_log_probs = []
        all_gate_actions = []

        for _ in range(self.n_comm_rounds):
            # a. Gate decision
            gate_logits = self.gate_head(h)  # (n, 2)
            gate_probs = F.softmax(gate_logits, dim=-1)

            if deterministic:
                gate_action = torch.argmax(gate_probs, dim=-1)  # (n,)
            else:
                gate_dist = torch.distributions.Categorical(gate_probs)
                gate_action = gate_dist.sample()  # (n,) ∈ {0, 1}

            gate_log_prob = torch.distributions.Categorical(gate_probs).log_prob(
                gate_action
            )  # (n,)

            all_gate_log_probs.append(gate_log_prob)
            all_gate_actions.append(gate_action)

            if not self._comm_active:
                # No-comm baseline or warmup phase: zero communication
                comm = torch.zeros(n, self.hidden_size, device=current_device)
            else:
                # b. Gated broadcast
                gate_mask = gate_action.unsqueeze(-1).float()  # (n, 1)
                gated_h = h * gate_mask  # (n, hidden_size)

                # c. Mean-pool messages from others (excluding self)
                # total_msgs = sum of all gated hidden states
                total_msgs = gated_h.sum(dim=0, keepdim=True)  # (1, hidden_size)
                # For each agent i: comm_i = (total - gated_h_i) / (n - 1)
                comm = (total_msgs - gated_h) / max(n - 1, 1)  # (n, hidden_size)

                # d. Transform communication
                comm = self.C_module(comm)  # (n, hidden_size)
                comm = self.comm_dropout(comm)

            # e. Additive combination (following official code)
            inp = x + comm  # (n, hidden_size)

            # f. LSTM update
            h, c = self.f_module(inp, (h, c))  # both (n, hidden_size)

        # 4. Compute action probabilities
        action_logits = self.action_head(h)  # (n, n_actions)
        action_probs = F.softmax(action_logits, dim=-1)  # (n, n_actions)

        # 5. Save persistent states
        self._hidden_states = h.detach()
        self._cell_states = c.detach()

        # Stack gate info: (n_agents, n_comm_rounds)
        gate_log_probs = torch.stack(all_gate_log_probs, dim=-1)
        gate_actions = torch.stack(all_gate_actions, dim=-1)

        # Store for buffer access
        self._last_gate_log_probs = gate_log_probs.detach()
        self._last_gate_actions = gate_actions.detach()

        return action_probs, gate_log_probs, gate_actions

    def act(
        self,
        all_obs_dicts: dict,
        reward: float | None = None,
        deterministic: bool = False,
        volunteer_threshold: float | None = None,
    ) -> tuple[dict, dict, torch.Tensor, torch.Tensor]:
        """Select actions for all agents using IC3Net communication.

        Args:
            all_obs_dicts: Dict {agent_id: obs_dict}
            reward: Current step reward
            deterministic: If True, use argmax for both gate and action
            volunteer_threshold: If set and deterministic, volunteer if p(vol) > threshold

        Returns:
            actions: Dict {agent_id: action_int}
            action_probs: Dict {agent_id: action_probs_tensor}
            gate_log_probs: Tensor (n_agents, n_comm_rounds)
            gate_actions: Tensor (n_agents, n_comm_rounds)
        """
        action_probs_all, gate_log_probs, gate_actions = self.forward(
            all_obs_dicts, reward, deterministic
        )

        actions = {}
        action_probs_dict = {}
        sorted_agent_ids = sorted(all_obs_dicts.keys())

        for idx, agent_id in enumerate(sorted_agent_ids):
            probs = action_probs_all[idx]
            if deterministic:
                if volunteer_threshold is not None:
                    action = 1 if probs[1].item() > volunteer_threshold else 0
                else:
                    action = torch.argmax(probs).item()
            else:
                action = torch.multinomial(probs, 1).item()
            actions[agent_id] = action
            action_probs_dict[agent_id] = probs

        return actions, action_probs_dict, gate_log_probs, gate_actions

    def forward_batch_pretensorized(
        self, x_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched forward pass for training with pre-tensorized observations.

        Re-runs communication from scratch (no persistent state) for proper gradients.

        Args:
            x_batch: Tensor (batch_size, n_agents, obs_dim+1) on device

        Returns:
            action_probs: (batch_size, n_agents, n_actions)
            gate_log_probs: (batch_size, n_agents, n_comm_rounds)
        """
        batch_size = x_batch.shape[0]
        n = self.n_agents
        device = x_batch.device

        # 1. Encode: (batch, n, obs_dim+1) → (batch, n, hidden_size)
        x = F.relu(self.encoder(x_batch))

        # 2. Initialize hidden states from encoded observations
        h = x.clone()  # (batch, n, hidden_size)
        c = torch.zeros(batch_size, n, self.hidden_size, device=device)

        # Get comm_mask on correct device: (n, n)
        comm_mask = self.comm_mask.to(device)

        # 3. Communication rounds
        all_gate_log_probs = []

        for _ in range(self.n_comm_rounds):
            # Reshape for gate computation
            h_flat = h.reshape(batch_size * n, self.hidden_size)

            # a. Gate decision (always sample during training for exploration)
            gate_logits = self.gate_head(h_flat)  # (batch*n, 2)
            gate_probs = F.softmax(gate_logits, dim=-1)
            gate_dist = torch.distributions.Categorical(gate_probs)
            gate_action = gate_dist.sample()  # (batch*n,)
            gate_log_prob = gate_dist.log_prob(gate_action)  # (batch*n,)

            all_gate_log_probs.append(
                gate_log_prob.reshape(batch_size, n)
            )  # (batch, n)

            if not self._comm_active:
                # No-comm baseline or warmup phase: zero communication
                comm = torch.zeros(batch_size, n, self.hidden_size, device=device)
            else:
                # b. Gated broadcast
                gate_mask = gate_action.reshape(batch_size, n, 1).float()  # (batch, n, 1)
                gated_h = h * gate_mask  # (batch, n, hidden_size)

                # c. Mean-pool messages from others using comm_mask
                # gated_h: (batch, n, hidden_size)
                # comm_mask: (n, n) — 1 for others, 0 for self
                # Expand for batch matmul: comm_mask (1, n, n) @ gated_h (batch, n, hidden)
                # Result: (batch, n, hidden) — each row is sum of others' gated hidden states
                comm = torch.bmm(
                    comm_mask.unsqueeze(0).expand(batch_size, -1, -1),  # (batch, n, n)
                    gated_h,  # (batch, n, hidden)
                ) / max(n - 1, 1)  # (batch, n, hidden)

                # d. Transform communication
                comm = self.C_module(comm.reshape(batch_size * n, self.hidden_size))
                comm = self.comm_dropout(comm)
                comm = comm.reshape(batch_size, n, self.hidden_size)

            # e. Additive combination
            inp = x + comm  # (batch, n, hidden_size)

            # f. LSTM update
            inp_flat = inp.reshape(batch_size * n, self.hidden_size)
            h_flat = h.reshape(batch_size * n, self.hidden_size)
            c_flat = c.reshape(batch_size * n, self.hidden_size)
            h_new, c_new = self.f_module(inp_flat, (h_flat, c_flat))
            h = h_new.reshape(batch_size, n, self.hidden_size)
            c = c_new.reshape(batch_size, n, self.hidden_size)

        # 4. Action probabilities
        h_flat = h.reshape(batch_size * n, self.hidden_size)
        action_logits = self.action_head(h_flat)  # (batch*n, n_actions)
        action_probs = F.softmax(action_logits, dim=-1)
        action_probs = action_probs.reshape(batch_size, n, -1)  # (batch, n, n_actions)

        # Stack gate log-probs: (batch, n, n_comm_rounds)
        gate_log_probs = torch.stack(all_gate_log_probs, dim=-1)

        return action_probs, gate_log_probs


class MLPCommActor(nn.Module):
    """MLP-based communication actor with parameter sharing.

    Same communication mechanism as IC3Net (gated mean-pool) but uses a
    stateless MLP backbone instead of LSTM. This allows a fair 2×2 comparison:

        |              | No Comm  | With Comm      |
        |--------------|----------|----------------|
        | MLP backbone | nocomm   | mlp_comm       |
        | LSTM backbone| ic3net_nocomm | ic3net    |

    Architecture:
    - encoder: obs → hidden_size
    - fc_hidden: hidden_size → hidden_size (processes obs + comm)
    - gate_head: hidden → 2-class gate logits
    - C_module: hidden → hidden (communication transform)
    - action_head: hidden → action logits

    No dropout is used, matching the IC3Net actor for fair comparison.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        n_agents: int,
        hidden_size: int = 64,
        n_comm_rounds: int = 2,
        comm_dropout: float = 0.0,
        weight_init: str = "xavier_uniform",
        comm_disabled: bool = False,
    ):
        super(MLPCommActor, self).__init__()
        self.n_agents = n_agents
        self.hidden_size = hidden_size
        self.n_comm_rounds = n_comm_rounds
        self.comm_disabled = comm_disabled

        # Calculate observation input size
        obs_input_dim = 0
        self.obs_keys = []
        for key, space in obs_space.items():
            self.obs_keys.append(key)
            if isinstance(space, spaces.Discrete):
                obs_input_dim += 1
            elif isinstance(space, spaces.Box):
                obs_input_dim += int(np.prod(space.shape))
            elif isinstance(space, spaces.MultiBinary):
                obs_input_dim += int(np.prod(space.shape))

        # +1 for reward
        self.obs_input_dim = obs_input_dim + 1

        # === Network modules ===
        # Encoder: obs+reward → hidden
        self.encoder = nn.Linear(self.obs_input_dim, hidden_size)

        # Gate head: hidden → 2 logits (communicate / don't communicate)
        self.gate_head = nn.Linear(hidden_size, 2)

        # Communication transform (shared across rounds)
        self.C_module = nn.Linear(hidden_size, hidden_size)

        # MLP hidden layer (replaces LSTMCell)
        # Takes (obs_encoding + comm) and produces updated hidden representation
        self.fc_hidden = nn.Linear(hidden_size, hidden_size)

        # Action head
        self.action_head = nn.Linear(hidden_size, action_space.n)

        # Communication dropout
        self.comm_dropout = nn.Dropout(p=comm_dropout)

        # Communication mask: prevents self-communication
        self.register_buffer(
            "comm_mask",
            torch.ones(n_agents, n_agents) - torch.eye(n_agents),
        )

        # Store last gate info for logging/buffer
        self._last_gate_log_probs: torch.Tensor | None = None
        self._last_gate_actions: torch.Tensor | None = None

        # Communication warmup: disable comm for first N episodes.
        # Set via set_comm_warmup() after construction.
        self._comm_warmup_episodes: int = 0
        self._current_episode: int = 0

        # Initialize weights
        self._init_weights(weight_init)

    def _init_weights(self, method: str = "xavier_uniform"):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if method == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif method == "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                elif method == "orthogonal":
                    nn.init.orthogonal_(module.weight)
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def set_comm_warmup(self, warmup_episodes: int) -> None:
        """Set number of warmup episodes during which communication is disabled."""
        self._comm_warmup_episodes = warmup_episodes

    def reset_hidden_states(self):
        """No-op for MLP (no persistent state). Keeps interface compatible.
        Also increments episode counter for comm warmup tracking."""
        self._current_episode += 1

    def _encode_observations(self, all_obs_dicts: dict, reward) -> torch.Tensor:
        """Encode observations for all agents into feature vectors."""
        current_device = next(self.parameters()).device
        agent_features = []

        for agent_id in sorted(all_obs_dicts.keys()):
            obs_dict = all_obs_dicts[agent_id]
            parts = []
            for key in self.obs_keys:
                if key in obs_dict:
                    val = obs_dict[key]
                    if isinstance(val, np.ndarray):
                        parts.append(
                            torch.from_numpy(val.flatten().astype(np.float32))
                        )
                    else:
                        parts.append(torch.tensor([val], dtype=torch.float32))

            if isinstance(reward, dict):
                r = reward.get(agent_id, 0.0)
            else:
                r = reward if reward is not None else 0.0
            parts.append(torch.tensor([r], dtype=torch.float32))
            agent_features.append(torch.cat(parts).to(current_device))

        return torch.stack(agent_features)  # (n_agents, obs_input_dim)

    def forward(
        self,
        all_obs_dicts: dict,
        reward: float | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with communication for all agents (rollout, CPU)."""
        current_device = next(self.parameters()).device
        n = self.n_agents

        # 1. Encode observations
        obs_flat = self._encode_observations(all_obs_dicts, reward)
        x = F.relu(self.encoder(obs_flat))  # (n, hidden_size)

        # 2. Start with encoded obs as hidden representation
        h = x  # (n, hidden_size)

        # 3. Communication rounds
        all_gate_log_probs = []
        all_gate_actions = []

        for _ in range(self.n_comm_rounds):
            # a. Gate decision
            gate_logits = self.gate_head(h)
            gate_probs = F.softmax(gate_logits, dim=-1)

            if deterministic:
                gate_action = torch.argmax(gate_probs, dim=-1)
            else:
                gate_dist = torch.distributions.Categorical(gate_probs)
                gate_action = gate_dist.sample()

            gate_log_prob = torch.distributions.Categorical(gate_probs).log_prob(gate_action)
            all_gate_log_probs.append(gate_log_prob)
            all_gate_actions.append(gate_action)

            warmup_active = self._current_episode < self._comm_warmup_episodes
            if self.comm_disabled or warmup_active:
                comm = torch.zeros(n, self.hidden_size, device=current_device)
            else:
                # b. Gated broadcast
                gate_mask = gate_action.unsqueeze(-1).float()
                gated_h = h * gate_mask

                # c. Mean-pool messages from others (excluding self)
                total_msgs = gated_h.sum(dim=0, keepdim=True)
                comm = (total_msgs - gated_h) / max(n - 1, 1)

                # d. Transform communication
                comm = self.C_module(comm)
                comm = self.comm_dropout(comm)

            # e. Additive combination + MLP update (replaces LSTM)
            inp = x + comm  # (n, hidden_size)
            h = F.relu(self.fc_hidden(inp))  # (n, hidden_size)

        # 4. Compute action probabilities
        action_logits = self.action_head(h)
        action_probs = F.softmax(action_logits, dim=-1)

        # Stack gate info
        gate_log_probs = torch.stack(all_gate_log_probs, dim=-1)
        gate_actions = torch.stack(all_gate_actions, dim=-1)

        # Store for buffer access
        self._last_gate_log_probs = gate_log_probs.detach()
        self._last_gate_actions = gate_actions.detach()

        return action_probs, gate_log_probs, gate_actions

    def act(
        self,
        all_obs_dicts: dict,
        reward: float | None = None,
        deterministic: bool = False,
        volunteer_threshold: float | None = None,
    ) -> tuple[dict, dict, torch.Tensor, torch.Tensor]:
        """Select actions for all agents."""
        action_probs_all, gate_log_probs, gate_actions = self.forward(
            all_obs_dicts, reward, deterministic
        )

        actions = {}
        action_probs_dict = {}
        sorted_agent_ids = sorted(all_obs_dicts.keys())

        for idx, agent_id in enumerate(sorted_agent_ids):
            probs = action_probs_all[idx]
            if deterministic:
                if volunteer_threshold is not None:
                    action = 1 if probs[1].item() > volunteer_threshold else 0
                else:
                    action = torch.argmax(probs).item()
            else:
                action = torch.multinomial(probs, 1).item()
            actions[agent_id] = action
            action_probs_dict[agent_id] = probs

        return actions, action_probs_dict, gate_log_probs, gate_actions

    def forward_batch_pretensorized(
        self, x_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched forward pass for training with pre-tensorized observations."""
        batch_size = x_batch.shape[0]
        n = self.n_agents
        device = x_batch.device

        # 1. Encode
        x = F.relu(self.encoder(x_batch))  # (batch, n, hidden_size)

        # 2. Start with encoded obs
        h = x  # (batch, n, hidden_size)

        # Get comm_mask on correct device
        comm_mask = self.comm_mask.to(device)

        # 3. Communication rounds
        all_gate_log_probs = []

        for _ in range(self.n_comm_rounds):
            h_flat = h.reshape(batch_size * n, self.hidden_size)

            # a. Gate decision
            gate_logits = self.gate_head(h_flat)
            gate_probs = F.softmax(gate_logits, dim=-1)
            gate_dist = torch.distributions.Categorical(gate_probs)
            gate_action = gate_dist.sample()
            gate_log_prob = gate_dist.log_prob(gate_action)

            all_gate_log_probs.append(gate_log_prob.reshape(batch_size, n))

            if self.comm_disabled:
                comm = torch.zeros(batch_size, n, self.hidden_size, device=device)
            else:
                # b. Gated broadcast
                gate_mask = gate_action.reshape(batch_size, n, 1).float()
                gated_h = h * gate_mask

                # c. Mean-pool messages from others using comm_mask
                comm = torch.bmm(
                    comm_mask.unsqueeze(0).expand(batch_size, -1, -1),
                    gated_h,
                ) / max(n - 1, 1)

                # d. Transform communication
                comm = self.C_module(comm.reshape(batch_size * n, self.hidden_size))
                comm = self.comm_dropout(comm)
                comm = comm.reshape(batch_size, n, self.hidden_size)

            # e. Additive combination + MLP update
            inp = x + comm
            h = F.relu(self.fc_hidden(inp.reshape(batch_size * n, self.hidden_size)))
            h = h.reshape(batch_size, n, self.hidden_size)

        # 4. Action probabilities
        h_flat = h.reshape(batch_size * n, self.hidden_size)
        action_logits = self.action_head(h_flat)
        action_probs = F.softmax(action_logits, dim=-1)
        action_probs = action_probs.reshape(batch_size, n, -1)

        gate_log_probs = torch.stack(all_gate_log_probs, dim=-1)

        return action_probs, gate_log_probs
