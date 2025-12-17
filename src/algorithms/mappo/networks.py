import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces


class ActorNetwork(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size=128, device=None):
        super(ActorNetwork, self).__init__()
        self.device = device if device is not None else torch.device("cpu")

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
        self.dropout = nn.Dropout(p=0.2)
        # Action head for discrete action space
        self.action_head = nn.Linear(hidden_size, action_space.n)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
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

    def act(self, obs, reward=None, deterministic=False):
        with torch.no_grad():
            action_probs = self(obs, reward)

            if deterministic:
                action = torch.argmax(action_probs).item()
            else:
                action = torch.multinomial(action_probs, 1).item()

            return action, action_probs

    def reset_history(self):
        """Reset the reward history buffer."""
        pass


class CriticNetwork(nn.Module):
    def __init__(self, obs_space, n_agents, hidden_size=256, device=None):
        super(CriticNetwork, self).__init__()
        self.device = device if device is not None else torch.device("cpu")

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
        self.dropout = nn.Dropout(p=0.2)
        self.value_head = nn.Linear(hidden_size, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
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
        reward_tensor = torch.tensor(
            [reward if reward is not None else 0.0],
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

    def reset_history(self):
        """Reset the reward history buffer."""
        pass
