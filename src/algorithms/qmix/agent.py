import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime

from ...core.display import print_colored


class AgentNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)


class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim=32):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hyper_w1 = nn.Linear(state_dim, n_agents * hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, agent_qs, state):
        bs = agent_qs.size(0)  # batch_size

        # Ensure agent_qs has correct shape [batch_size, n_agents]
        if len(agent_qs.shape) == 1:
            agent_qs = agent_qs.unsqueeze(0)  # Add batch dimension if missing

        # Generate hypernetwork weights and biases
        w1 = torch.abs(self.hyper_w1(state))  # [batch_size, n_agents * hidden_dim]
        w1 = w1.view(bs, self.n_agents, -1)  # [batch_size, n_agents, hidden_dim]

        b1 = self.hyper_b1(state)  # [batch_size, hidden_dim]
        b1 = b1.view(bs, 1, -1)  # [batch_size, 1, hidden_dim]

        # First layer: agent_qs * w1 + b1
        # agent_qs: [batch_size, n_agents] -> [batch_size, 1, n_agents] for bmm
        # w1: [batch_size, n_agents, hidden_dim]
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1)  # [batch_size, 1, hidden_dim]
        hidden = hidden.squeeze(1) + b1.squeeze(1)  # [batch_size, hidden_dim]
        hidden = nn.functional.elu(hidden)

        # Second layer
        w2 = torch.abs(self.hyper_w2(state))  # [batch_size, hidden_dim]
        w2 = w2.view(bs, -1, 1)  # [batch_size, hidden_dim, 1]

        b2 = self.hyper_b2(state)  # [batch_size, 1]

        # hidden: [batch_size, hidden_dim] -> [batch_size, 1, hidden_dim] for bmm
        # w2: [batch_size, hidden_dim, 1]
        y = torch.bmm(hidden.unsqueeze(1), w2).squeeze(-1)  # [batch_size, 1]
        y = y + b2  # [batch_size, 1]

        return y


class QMIXAgent:
    def __init__(
        self,
        env,
        n_agents=None,
        state_dim=None,
        obs_dim=None,
        n_actions=None,
        device=None,
        **kwargs,
    ):
        self.env = env
        self.n_agents = n_agents or len(env.agents)

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

        print_colored(f"QMIXAgent using device: {self.device}", "blue")

        # Calculate observation dimension from the environment's observation space
        first_agent = env.agents[0]
        obs_space = env.observation_space(first_agent.id)
        self.obs_dim = self._calculate_obs_dim(obs_space)

        # Calculate state dimension - add environment-specific global state features
        # Global state includes: all agent observations + global environment info
        global_state_features = 5  # Number of future cases, pending cases, completed cases, current time step, upcoming task info
        self.state_dim = state_dim or (
            self.obs_dim * self.n_agents + global_state_features
        )

        self.n_actions = n_actions or env.action_space(env.agents[0]).n

        # Create networks
        self.agent_net = AgentNetwork(self.obs_dim, self.n_actions).to(self.device)
        self.target_agent_net = AgentNetwork(self.obs_dim, self.n_actions).to(
            self.device
        )
        self.mixing_net = MixingNetwork(self.n_agents, self.state_dim).to(self.device)
        self.target_mixing_net = MixingNetwork(self.n_agents, self.state_dim).to(
            self.device
        )

        # Setup optimizer
        self.optimizer = optim.Adam(
            list(self.agent_net.parameters()) + list(self.mixing_net.parameters()),
            lr=kwargs.get("lr", 0.0005),
        )

        # Hyperparameters
        self.gamma = kwargs.get("gamma", 0.99)
        self.epsilon = kwargs.get("epsilon", 0.05)
        self.epsilon_decay = kwargs.get("epsilon_decay", 0.995)
        self.epsilon_min = kwargs.get("epsilon_min", 0.01)

        # Initialize target networks
        self.update_target()

        # Print initialization summary
        print_colored("QMIX Agent Initialized", "green")
        print_colored(f"  Agents: {self.n_agents}", "cyan")
        print_colored(f"  Observation dim: {self.obs_dim}", "cyan")
        print_colored(f"  State dim: {self.state_dim}", "cyan")
        print_colored(f"  Actions: {self.n_actions}", "cyan")
        print_colored(f"  Learning rate: {kwargs.get('lr', 0.0005)}", "cyan")
        print_colored(f"  Gamma: {self.gamma}", "cyan")
        print_colored(f"  Initial epsilon: {self.epsilon}", "cyan")

    def _calculate_obs_dim(self, obs_space):
        """Calculate the total observation dimension from observation space"""
        total_dim = 0

        # Handle Dict observation space (PettingZoo style)
        if hasattr(obs_space, "items"):
            for key, space in obs_space.items():
                if hasattr(space, "n"):  # Discrete space
                    total_dim += 1
                elif hasattr(space, "shape"):  # Box or MultiBinary space
                    total_dim += int(np.prod(space.shape))
                else:
                    total_dim += 1  # Fallback
        # Handle simple observation space (Gym style)
        elif hasattr(obs_space, "shape"):
            total_dim = int(np.prod(obs_space.shape))
        elif hasattr(obs_space, "n"):
            total_dim = obs_space.n
        else:
            # Fallback - try to get a reasonable default
            total_dim = 10

        return total_dim

    def _flatten_observation(self, obs_dict):
        """Convert dictionary observation to flat numpy array"""
        obs_list = []
        for key in sorted(obs_dict.keys()):  # Sort keys for consistent ordering
            value = obs_dict[key]
            if isinstance(value, (int, float)):
                obs_list.append(float(value))
            elif isinstance(value, (list, np.ndarray)):
                obs_list.extend([float(x) for x in np.array(value).flatten()])
            else:
                # Try to convert to float, fallback to 0.0
                try:
                    obs_list.append(float(value))
                except (TypeError, ValueError):
                    obs_list.append(0.0)
        return np.array(obs_list)

    def select_actions(self, observations, deterministic=False):
        """Select actions for all agents with optional logging."""
        # Convert dictionary observations to flat arrays
        obs_arrays = []
        for agent in self.env.agents:
            flat_obs = self._flatten_observation(observations[agent.id])
            obs_arrays.append(flat_obs)

        obs = np.stack(obs_arrays)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            q_values = self.agent_net(obs)

        if deterministic or np.random.rand() > self.epsilon:
            actions = q_values.argmax(dim=-1).cpu().numpy()
        else:
            actions = np.random.randint(self.n_actions, size=self.n_agents)

        # Decay epsilon for exploration (only during training)
        if not deterministic:
            old_epsilon = self.epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Log epsilon decay occasionally
            if old_epsilon != self.epsilon and hasattr(self, "_last_epsilon_log"):
                if (
                    abs(old_epsilon - self._last_epsilon_log) >= 0.01
                ):  # Log every 1% change
                    print_colored(
                        f"Epsilon decayed: {old_epsilon:.4f} -> {self.epsilon:.4f}",
                        "purple",
                    )
                    self._last_epsilon_log = self.epsilon
            elif not hasattr(self, "_last_epsilon_log"):
                self._last_epsilon_log = self.epsilon

        return {agent.id: actions[i] for i, agent in enumerate(self.env.agents)}, None

    def get_training_summary(self) -> str:
        """Get a summary of the current training state."""
        summary = []
        summary.append("QMIX Agent Training State")
        summary.append("=" * 26)
        summary.append(f"Device: {self.device}")
        summary.append(f"Current epsilon: {self.epsilon:.4f}")
        summary.append(f"Epsilon decay: {self.epsilon_decay}")
        summary.append(f"Epsilon minimum: {self.epsilon_min}")
        summary.append(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        summary.append(f"Gamma (discount): {self.gamma}")
        summary.append("")

        # Network parameter counts
        agent_params = sum(
            p.numel() for p in self.agent_net.parameters() if p.requires_grad
        )
        mixing_params = sum(
            p.numel() for p in self.mixing_net.parameters() if p.requires_grad
        )
        total_params = agent_params + mixing_params

        summary.append("Network Parameters:")
        summary.append(f"  Agent Network: {agent_params:,}")
        summary.append(f"  Mixing Network: {mixing_params:,}")
        summary.append(f"  Total: {total_params:,}")
        summary.append("")

        return "\n".join(summary)

    def update_target(self):
        """Update target networks with enhanced logging."""
        print_colored("Updating QMIX target networks...", "purple")
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())
        print_colored("Target networks updated successfully", "purple")

    def get_q_values(self, observations):
        """Get Q-values for given observations"""
        obs_arrays = []
        for agent in self.env.agents:
            flat_obs = self._flatten_observation(observations[agent.id])
            obs_arrays.append(flat_obs)

        obs = np.stack(obs_arrays)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        return self.agent_net(obs)

    def get_global_state(self, observations):
        """Create global state from all agent observations plus environment info"""
        obs_arrays = []
        for agent in self.env.agents:
            flat_obs = self._flatten_observation(observations[agent.id])
            obs_arrays.append(flat_obs)

        # Combine all agent observations
        global_obs = np.concatenate(obs_arrays)

        # Add global environment features
        env_features = np.array(
            [
                len(self.env.future_cases),  # Number of future cases
                len(self.env.pending_cases),  # Number of pending cases
                len(self.env.completed_cases),  # Number of completed cases
                self.env.steps,  # Current step
                (
                    1.0 if self.env.upcoming_case is not None else 0.0
                ),  # Whether there's an upcoming case
            ]
        )

        return np.concatenate([global_obs, env_features])

    def save_models(self, path):
        """Save model weights to the specified path with comprehensive logging."""
        print_colored(f"Saving QMIX models to: {path}", "cyan")

        # Ensure directory exists
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )

        try:
            # Save all model states and hyperparameters
            checkpoint = {
                "agent_net": self.agent_net.state_dict(),
                "mixing_net": self.mixing_net.state_dict(),
                "target_agent_net": self.target_agent_net.state_dict(),
                "target_mixing_net": self.target_mixing_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "hyperparameters": {
                    "n_agents": self.n_agents,
                    "obs_dim": self.obs_dim,
                    "state_dim": self.state_dim,
                    "n_actions": self.n_actions,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "epsilon_decay": self.epsilon_decay,
                    "epsilon_min": self.epsilon_min,
                    "lr": self.optimizer.param_groups[0]["lr"],
                },
                "device": str(self.device),
            }

            torch.save(checkpoint, path)
            print_colored(f"Successfully saved QMIX models to: {path}", "green")

        except Exception as e:
            print_colored(f"Error saving QMIX models: {str(e)}", "red")
            raise

    def load_models(self, path):
        """Load model weights from the specified path with comprehensive logging."""
        print_colored(f"Loading QMIX models from: {path}", "cyan")

        try:
            checkpoint = torch.load(path, map_location=self.device)

            # Load model states
            self.agent_net.load_state_dict(checkpoint["agent_net"])
            self.mixing_net.load_state_dict(checkpoint["mixing_net"])

            # Load target networks if available
            if "target_agent_net" in checkpoint:
                self.target_agent_net.load_state_dict(checkpoint["target_agent_net"])
            else:
                self.update_target()  # Fallback: copy from main networks

            if "target_mixing_net" in checkpoint:
                self.target_mixing_net.load_state_dict(checkpoint["target_mixing_net"])

            # Load optimizer state if available
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])

            # Load hyperparameters if available
            if "hyperparameters" in checkpoint:
                hyperparams = checkpoint["hyperparameters"]
                self.gamma = hyperparams.get("gamma", self.gamma)
                self.epsilon = hyperparams.get("epsilon", self.epsilon)
                self.epsilon_decay = hyperparams.get(
                    "epsilon_decay", self.epsilon_decay
                )
                self.epsilon_min = hyperparams.get("epsilon_min", self.epsilon_min)

                print_colored("Loaded hyperparameters:", "cyan")
                print_colored(f"  Gamma: {self.gamma}", "cyan")
                print_colored(f"  Epsilon: {self.epsilon}", "cyan")
                print_colored(
                    f"  Learning rate: {hyperparams.get('lr', 'unknown')}", "cyan"
                )

            print_colored(f"Successfully loaded QMIX models from: {path}", "green")

        except Exception as e:
            print_colored(f"Error loading QMIX models: {str(e)}", "red")
            raise
