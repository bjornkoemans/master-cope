import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces

# Default hyperparameters
DROPOUT_RATE = 0.2
WEIGHT_INIT = "xavier_uniform"


def pretensorize_actor_observations(obs_dicts, rewards, obs_keys):
    """Convert a list of observation dicts + rewards into a single pre-stacked tensor.

    Args:
        obs_dicts: List of observation dicts (one per timestep), each {obs_key: value}
        rewards: List of scalar rewards (one per timestep)
        obs_keys: List of observation key names in order

    Returns:
        torch.Tensor of shape (N, feature_dim) on CPU, dtype float32
    """
    n = len(obs_dicts)
    # Compute feature layout from first sample
    first = obs_dicts[0]
    feature_layout = []  # list of (key, size)
    total_dim = 0
    for key in obs_keys:
        if key in first:
            val = first[key]
            size = val.size if isinstance(val, np.ndarray) else 1
            feature_layout.append((key, size))
            total_dim += size
    total_dim += 1  # reward

    # Pre-allocate and fill in-place
    result = np.empty((n, total_dim), dtype=np.float32)
    for i in range(n):
        obs_dict = obs_dicts[i]
        col = 0
        for key, size in feature_layout:
            val = obs_dict[key]
            if size > 1:
                result[i, col:col + size] = val.flat
            else:
                result[i, col] = float(val) if isinstance(val, np.ndarray) else val
            col += size
        result[i, col] = rewards[i] if rewards is not None else 0.0

    return torch.from_numpy(result)


def pretensorize_critic_observations(obs_lists, rewards, obs_keys):
    """Convert a list of per-timestep agent observation lists + rewards into a single tensor.

    Args:
        obs_lists: List of lists, where obs_lists[i] is a list of agent obs dicts
        rewards: List of scalar rewards (one per timestep)
        obs_keys: List of observation key names in order

    Returns:
        torch.Tensor of shape (N, n_agents * single_agent_dim + 1) on CPU, dtype float32
    """
    n = len(obs_lists)
    n_agents = len(obs_lists[0])

    # Compute feature layout from first agent's obs
    first_agent_obs = obs_lists[0][0]
    feature_layout = []  # list of (key, size)
    agent_dim = 0
    for key in obs_keys:
        if key in first_agent_obs:
            val = first_agent_obs[key]
            size = val.size if isinstance(val, np.ndarray) else 1
            feature_layout.append((key, size))
            agent_dim += size
    total_dim = n_agents * agent_dim + 1  # +1 for reward

    # Pre-allocate and fill in-place
    result = np.empty((n, total_dim), dtype=np.float32)
    for i in range(n):
        agent_obs_list = obs_lists[i]
        col = 0
        for agent_obs in agent_obs_list:
            for key, size in feature_layout:
                val = agent_obs[key]
                if size > 1:
                    result[i, col:col + size] = val.flat
                else:
                    result[i, col] = float(val) if isinstance(val, np.ndarray) else val
                col += size
        result[i, col] = rewards[i] if rewards is not None else 0.0

    return torch.from_numpy(result)


class ActorNetwork(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size=128, dropout_rate=None, weight_init=None, device=None):
        super(ActorNetwork, self).__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.dropout_rate = dropout_rate if dropout_rate is not None else DROPOUT_RATE
        self.weight_init = weight_init if weight_init is not None else WEIGHT_INIT

        # Calculate input size from observation space
        input_size = 0
        self.obs_keys = []

        # Process Dict observation space
        for key, space in obs_space.items():
            self.obs_keys.append(key)
            if isinstance(space, spaces.Discrete):
                input_size += 1
            elif isinstance(space, spaces.Box):
                input_size += int(np.prod(space.shape))
            elif isinstance(space, spaces.MultiBinary):
                input_size += int(np.prod(space.shape))

        # Add reward history to input size
        input_size += 1  # For current reward

        # Deeper network layers with dropout
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        # Action head for discrete action space
        self.action_head = nn.Linear(hidden_size, action_space.n)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using configured initialization method.

        The action_head (output layer) is initialized near-zero so that the
        initial softmax produces ~uniform probabilities (50/50 for 2 actions).
        This prevents early policy collapse where one agent dominates due to
        random initialization asymmetry.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module is self.action_head:
                    # Near-zero init for action head → softmax([~0, ~0]) ≈ [0.5, 0.5]
                    nn.init.uniform_(module.weight, -0.01, 0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    if self.weight_init == "xavier_uniform":
                        nn.init.xavier_uniform_(module.weight)
                    elif self.weight_init == "xavier_normal":
                        nn.init.xavier_normal_(module.weight)
                    elif self.weight_init == "kaiming_uniform":
                        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                    elif self.weight_init == "kaiming_normal":
                        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    elif self.weight_init == "orthogonal":
                        nn.init.orthogonal_(module.weight)
                    else:
                        nn.init.xavier_uniform_(module.weight)  # fallback

                    if module.bias is not None:
                        nn.init.normal_(module.bias, 0, 0.01)

    def forward(self, obs_dict, reward=None):
        # Process observation dictionary into a flat vector
        x_parts = []

        # Get the current device of the model
        current_device = next(self.parameters()).device

        for key in self.obs_keys:
            if key in obs_dict:
                # Handle different observation components
                if isinstance(obs_dict[key], np.ndarray):
                    # Use from_numpy for better performance, then move to device
                    tensor_data = torch.from_numpy(
                        obs_dict[key].flatten().astype(np.float32)
                    )
                    x_parts.append(tensor_data.to(current_device))
                else:
                    # Handle scalar values or other types
                    x_parts.append(
                        torch.tensor(
                            [obs_dict[key]], device=current_device, dtype=torch.float32
                        )
                    )

        # Add current reward to input (create as single tensor)
        reward_tensor = torch.tensor(
            [reward if reward is not None else 0.0],
            device=current_device,
            dtype=torch.float32,
        )
        x_parts.append(reward_tensor)

        x = torch.cat(x_parts)

        # Process through deeper network with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        action_probs = F.softmax(self.action_head(x), dim=-1)

        return action_probs

    def forward_batch(self, obs_dicts, rewards=None):
        """Batch forward pass for multiple observations."""
        # Get the current device of the model
        current_device = next(self.parameters()).device

        # Pre-allocate lists for better performance
        batch_inputs = []

        # Process all observations efficiently
        for i, obs_dict in enumerate(obs_dicts):
            x_parts = []

            for key in self.obs_keys:
                if key in obs_dict:
                    # Handle different observation components
                    if isinstance(obs_dict[key], np.ndarray):
                        # Convert to tensor efficiently
                        tensor_data = torch.from_numpy(
                            obs_dict[key].flatten().astype(np.float32)
                        )
                        x_parts.append(tensor_data)
                    else:
                        x_parts.append(
                            torch.tensor([obs_dict[key]], dtype=torch.float32)
                        )

            # Add current reward to input
            reward = rewards[i] if rewards is not None else 0.0
            x_parts.append(torch.tensor([reward], dtype=torch.float32))

            # Concatenate and add to batch (still on CPU)
            batch_inputs.append(torch.cat(x_parts))

        # Move entire batch to device at once
        x_batch = torch.stack(batch_inputs).to(current_device)

        # Process through deeper network with dropout (batch processing)
        x = F.relu(self.fc1(x_batch))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        action_probs = F.softmax(self.action_head(x), dim=-1)

        return action_probs

    def forward_batch_pretensorized(self, x_batch):
        """Batch forward pass using pre-tensorized input already on device.

        Args:
            x_batch: Tensor of shape (batch_size, input_dim) already on the correct device

        Returns:
            Action probabilities tensor of shape (batch_size, n_actions)
        """
        x = F.relu(self.fc1(x_batch))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        action_probs = F.softmax(self.action_head(x), dim=-1)
        return action_probs

    def act(self, obs, reward=None, deterministic=False, volunteer_threshold=None):
        with torch.no_grad():
            action_probs = self(obs, reward)

            if deterministic:
                if volunteer_threshold is not None:
                    # Use threshold: action=1 (volunteer) if p(vol) > threshold
                    action = 1 if action_probs[1].item() > volunteer_threshold else 0
                else:
                    action = torch.argmax(action_probs).item()
            else:
                action = torch.multinomial(action_probs, 1).item()

            return action, action_probs

    def reset_history(self):
        """Reset the reward history buffer."""
        pass


class CriticNetwork(nn.Module):
    def __init__(self, obs_space, n_agents, hidden_size=256, dropout_rate=None, weight_init=None, device=None):
        super(CriticNetwork, self).__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.dropout_rate = dropout_rate if dropout_rate is not None else DROPOUT_RATE
        self.weight_init = weight_init if weight_init is not None else WEIGHT_INIT

        # For centralized critic, we combine observations from all agents
        # and potentially global state information

        # Calculate input size from observation space (for all agents)
        single_agent_obs_size = 0
        self.obs_keys = []

        # Process Dict observation space
        for key, space in obs_space.items():
            self.obs_keys.append(key)
            if isinstance(space, spaces.Discrete):
                single_agent_obs_size += 1  # One-hot encoding
            elif isinstance(space, spaces.Box):
                single_agent_obs_size += int(np.prod(space.shape))
            elif isinstance(space, spaces.MultiBinary):
                single_agent_obs_size += int(np.prod(space.shape))

        # Total input size = single agent obs size * number of agents + reward
        input_size = single_agent_obs_size * n_agents + 1  # +1 for reward

        # Deeper network layers with dropout
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.value_head = nn.Linear(hidden_size, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using configured initialization method."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.weight_init == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif self.weight_init == "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                elif self.weight_init == "kaiming_uniform":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                elif self.weight_init == "kaiming_normal":
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                elif self.weight_init == "orthogonal":
                    nn.init.orthogonal_(module.weight)
                else:
                    nn.init.xavier_uniform_(module.weight)  # fallback

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, obs_dicts, reward=None):
        # Process and concatenate observations from all agents
        all_agent_inputs = []

        # Get the current device of the model
        current_device = next(self.parameters()).device

        for agent_obs in obs_dicts:
            agent_parts = []
            for key in self.obs_keys:
                if key in agent_obs:
                    # Handle different observation components
                    if isinstance(agent_obs[key], np.ndarray):
                        # Use from_numpy for better performance
                        tensor_data = torch.from_numpy(
                            agent_obs[key].flatten().astype(np.float32)
                        )
                        agent_parts.append(tensor_data.to(current_device))
                    else:
                        # Handle scalar values
                        agent_parts.append(
                            torch.tensor(
                                [agent_obs[key]],
                                device=current_device,
                                dtype=torch.float32,
                            )
                        )

            if agent_parts:
                agent_input = torch.cat(agent_parts)
                all_agent_inputs.append(agent_input)

        # Concatenate all agents' observations
        x = torch.cat(all_agent_inputs)

        # Add current reward to input (single tensor creation)
        # If reward is a per-agent dict, use the mean for the centralized critic
        if isinstance(reward, dict):
            r_val = float(np.mean(list(reward.values()))) if reward else 0.0
        else:
            r_val = reward if reward is not None else 0.0
        reward_tensor = torch.tensor(
            [r_val],
            device=current_device,
            dtype=torch.float32,
        )
        x = torch.cat([x, reward_tensor])

        # Process through deeper network with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        value = self.value_head(x)

        return value

    def forward_batch(self, obs_dicts_batch, rewards=None):
        """Batch forward pass for multiple observation sets."""
        # Get the current device of the model
        current_device = next(self.parameters()).device

        # Process all observation sets in batch
        batch_inputs = []

        for i, obs_dicts in enumerate(obs_dicts_batch):
            # Process and concatenate observations from all agents for this sample
            all_agent_inputs = []

            for agent_obs in obs_dicts:
                agent_parts = []
                for key in self.obs_keys:
                    if key in agent_obs:
                        # Handle different observation components
                        if isinstance(agent_obs[key], np.ndarray):
                            tensor_data = torch.from_numpy(
                                agent_obs[key].flatten().astype(np.float32)
                            )
                            agent_parts.append(tensor_data)
                        else:
                            agent_parts.append(
                                torch.tensor([agent_obs[key]], dtype=torch.float32)
                            )

                if agent_parts:
                    agent_input = torch.cat(agent_parts)
                    all_agent_inputs.append(agent_input)

            # Concatenate all agents' observations for this sample
            x = torch.cat(all_agent_inputs)

            # Add current reward to input
            reward = rewards[i] if rewards is not None else 0.0
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            x = torch.cat([x, reward_tensor])

            batch_inputs.append(x)

        # Move entire batch to device at once and process
        x_batch = torch.stack(batch_inputs).to(current_device)

        # Process through deeper network with dropout (batch processing)
        x = F.relu(self.fc1(x_batch))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        values = self.value_head(x)

        return values

    def forward_batch_pretensorized(self, x_batch):
        """Batch forward pass using pre-tensorized input already on device.

        Args:
            x_batch: Tensor of shape (batch_size, input_dim) already on the correct device

        Returns:
            Value predictions tensor of shape (batch_size, 1)
        """
        x = F.relu(self.fc1(x_batch))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        values = self.value_head(x)
        return values

    def reset_history(self):
        """Reset the reward history buffer."""
        pass


class QNetwork(nn.Module):
    """COMA-style centralized Q-network: Q(s, a_1, ..., a_n).

    Takes joint observations (all agents) + joint actions as input.
    Used for computing counterfactual baselines per agent.

    The Q-value represents the expected return given the joint state
    and the joint action vector. By marginalizing over agent i's action
    while keeping others fixed, we get the counterfactual baseline b_i.
    """

    def __init__(self, obs_space, n_agents, n_actions=2, hidden_size=256,
                 dropout_rate=None, weight_init=None, device=None):
        super(QNetwork, self).__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.dropout_rate = dropout_rate if dropout_rate is not None else DROPOUT_RATE
        self.weight_init = weight_init if weight_init is not None else WEIGHT_INIT
        self.n_agents = n_agents
        self.n_actions = n_actions

        # Calculate input size from observation space (all agents)
        single_agent_obs_size = 0
        self.obs_keys = []
        for key, space in obs_space.items():
            self.obs_keys.append(key)
            if isinstance(space, spaces.Discrete):
                single_agent_obs_size += 1
            elif isinstance(space, spaces.Box):
                single_agent_obs_size += int(np.prod(space.shape))
            elif isinstance(space, spaces.MultiBinary):
                single_agent_obs_size += int(np.prod(space.shape))

        # Input = joint observations + reward + joint actions (one-hot per agent)
        input_size = single_agent_obs_size * n_agents + 1 + n_agents * n_actions

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.q_head = nn.Linear(hidden_size, 1)  # Single Q-value output

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.weight_init == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif self.weight_init == "orthogonal":
                    nn.init.orthogonal_(module.weight)
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward_batch(self, obs_tensor, action_onehots):
        """Batch forward pass.

        Args:
            obs_tensor: (batch, joint_obs_dim + 1) — joint observations + reward
                        (same format as CriticNetwork pretensorized input)
            action_onehots: (batch, n_agents * n_actions) — one-hot encoded joint actions

        Returns:
            Q-values: (batch, 1)
        """
        x = torch.cat([obs_tensor, action_onehots], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        return self.q_head(x)

    def reset_history(self):
        pass
