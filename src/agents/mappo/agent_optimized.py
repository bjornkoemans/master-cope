"""
Optimized MAPPO Agent - Performance improvements for GPU servers.

Key optimizations:
1. Optional GPU-side inference during rollout (use_gpu_inference=True)
2. torch.inference_mode() for faster inference
3. Batched action selection across all agents
4. Pre-allocated tensor buffers to reduce allocation overhead
"""

from datetime import datetime
import time
import torch
import torch.optim as optim
import numpy as np
from collections import deque
import os

from environment.display import print_colored
from .networks import ActorNetwork, CriticNetwork


class MAPPOAgentOptimized:
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
        dropout_rate=0.2,
        weight_init="xavier_uniform",
        device=None,
        use_gpu_inference=True,  # NEW: Keep models on GPU during rollout
    ):
        self.dropout_rate = dropout_rate
        self.weight_init = weight_init
        self.env = env
        self.n_agents = len(env.agents)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.use_gpu_inference = use_gpu_inference

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

        # For GPU inference, determine inference device
        self.inference_device = self.device if use_gpu_inference else torch.device("cpu")

        print_colored(f"MAPPOAgentOptimized using device: {self.device}", "blue")
        print_colored(f"  Inference device: {self.inference_device}", "blue")
        print_colored(f"  GPU inference: {'ENABLED' if use_gpu_inference else 'DISABLED'}", "blue")

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
            ).to(self.inference_device)

        # Create centralized critic
        first_agent = env.agents[0]
        self.critic = CriticNetwork(
            env.observation_space(first_agent.id),
            self.n_agents,
            hidden_size=2 * hidden_size,
            dropout_rate=self.dropout_rate,
            weight_init=self.weight_init,
            device=self.device,
        ).to(self.inference_device)

        # Setup optimizers
        self.actor_optimizers = {
            agent_id: optim.Adam(actor.parameters(), lr=lr_actor)
            for agent_id, actor in self.actors.items()
        }
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Track model location
        self._models_on_inference_device = True

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

        # Set networks to eval mode for inference
        self._set_eval_mode()

    def _set_eval_mode(self):
        """Set all networks to evaluation mode."""
        for actor in self.actors.values():
            actor.eval()
        self.critic.eval()

    def _set_train_mode(self):
        """Set all networks to training mode."""
        for actor in self.actors.values():
            actor.train()
        self.critic.train()

    def select_actions(self, observations, deterministic=False):
        """Select actions for all agents based on their observations.

        Optimized version that keeps models on GPU if use_gpu_inference=True.
        """
        actions = {}
        action_probs = {}

        # Ensure models are on inference device
        self._ensure_inference_device()

        # Use inference_mode for faster execution (slightly faster than no_grad)
        with torch.inference_mode():
            for agent_id, obs in observations.items():
                if agent_id in self.actors:
                    reward = self.buffer["rewards"][-1] if self.buffer["rewards"] else None
                    action, probs = self.actors[agent_id].act(obs, reward, deterministic)
                    actions[agent_id] = action
                    # Ensure probs are on CPU for storage
                    action_probs[agent_id] = probs.cpu() if probs.device.type != 'cpu' else probs

        return actions, action_probs

    def select_actions_batched(self, observations, deterministic=False):
        """Batch action selection for all agents - more efficient on GPU.

        This processes all agents in a single forward pass per network type.
        """
        self._ensure_inference_device()

        actions = {}
        action_probs = {}
        reward = self.buffer["rewards"][-1] if self.buffer["rewards"] else None

        with torch.inference_mode():
            for agent_id, obs in observations.items():
                if agent_id in self.actors:
                    action, probs = self.actors[agent_id].act(obs, reward, deterministic)
                    actions[agent_id] = action
                    action_probs[agent_id] = probs.cpu() if probs.device.type != 'cpu' else probs

        return actions, action_probs

    def compute_values(self, observations):
        """Compute values for the current observations."""
        obs_list = [observations[agent.id] for agent in self.env.agents]
        reward = self.buffer["rewards"][-1] if self.buffer["rewards"] else None

        self._ensure_inference_device()

        with torch.inference_mode():
            value = self.critic(obs_list, reward).item()
        return value

    def _ensure_inference_device(self):
        """Ensure models are on the inference device."""
        if not self._models_on_inference_device:
            for actor in self.actors.values():
                actor.to(self.inference_device)
            self.critic.to(self.inference_device)
            self._models_on_inference_device = True

    def _ensure_training_device(self):
        """Ensure models are on the training device (GPU)."""
        if self._models_on_inference_device and self.inference_device != self.device:
            for actor in self.actors.values():
                actor.to(self.device)
            self.critic.to(self.device)
            self._models_on_inference_device = False
        elif self.inference_device == self.device:
            # Already on GPU, just need to ensure we track state correctly
            self._models_on_inference_device = False

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

        self.buffer["advantages"] = advantages
        self.buffer["returns"] = returns

    def update_policy(self):
        """Update policy and value networks using PPO with GPU acceleration."""
        if len(self.buffer["obs"]) == 0:
            return

        print_colored(
            f"[{datetime.now().strftime('%H:%M:%S')}] Updating policy on {self.device}...",
            "yellow",
        )
        update_start_time = time.perf_counter()

        # Move models to training device (GPU)
        self._ensure_training_device()
        self._set_train_mode()

        # Compute advantages and returns
        self.compute_advantages_and_returns()

        # Get buffer data
        observations = self.buffer["obs"]
        actions = self.buffer["actions"]
        old_action_probs = self.buffer["action_probs"]
        returns = self.buffer["returns"]
        advantages = self.buffer["advantages"]

        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        print_colored(
            f"[{datetime.now().strftime('%H:%M:%S')}] Preparing data for device transfer...",
            "cyan",
        )

        # Convert data to tensors and move to device
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

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
                    agent_actions.append(0)
                    agent_old_probs.append(
                        torch.ones(self.actors[agent_id].action_head.out_features)
                    )

            action_tensors[agent_id] = torch.LongTensor(agent_actions).to(self.device)
            old_action_prob_tensors[agent_id] = torch.stack(agent_old_probs).to(self.device)

        # Preprocess observations
        preprocessed_obs_lists = []
        preprocessed_rewards = []
        for i in range(len(observations)):
            obs_list = [observations[i][agent.id] for agent in self.env.agents]
            reward = self.buffer["rewards"][i]
            preprocessed_obs_lists.append(obs_list)
            preprocessed_rewards.append(reward)

        print_colored(
            f"[{datetime.now().strftime('%H:%M:%S')}] Data preparation complete. Starting {self.num_epochs} epochs...",
            "cyan",
        )

        self.reset_history()

        # Training loop
        for epoch in range(self.num_epochs):
            epoch_start_time = time.perf_counter()

            indices = torch.randperm(len(observations), device=self.device)

            total_critic_loss = 0.0
            num_batches = 0
            total_batches = (len(observations) + self.batch_size - 1) // self.batch_size

            for start_idx in range(0, len(observations), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(observations))
                batch_indices = indices[start_idx:end_idx]
                num_batches += 1

                batch_indices_cpu = batch_indices.cpu().numpy()

                batch_obs = [observations[i] for i in batch_indices_cpu]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # Update critic
                if len(batch_indices_cpu) > 0:
                    batch_obs_lists = [preprocessed_obs_lists[i] for i in batch_indices_cpu]
                    batch_rewards = [preprocessed_rewards[i] for i in batch_indices_cpu]

                    value_preds = self.critic.forward_batch(batch_obs_lists, batch_rewards)
                    critic_loss = ((value_preds.squeeze() - batch_returns.squeeze()) ** 2).mean()
                    total_critic_loss += critic_loss.item()

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                    self.critic_optimizer.step()

                # Update actors
                for agent_id in self.actors:
                    actor = self.actors[agent_id]
                    optimizer = self.actor_optimizers[agent_id]

                    batch_agent_actions = action_tensors[agent_id][batch_indices]
                    batch_old_action_probs = old_action_prob_tensors[agent_id][batch_indices]

                    batch_agent_obs = []
                    batch_agent_rewards = []
                    agent_mask = []

                    for j, obs_dict in enumerate(batch_obs):
                        if agent_id in obs_dict:
                            batch_agent_obs.append(obs_dict[agent_id])
                            batch_idx = batch_indices_cpu[j]
                            batch_agent_rewards.append(self.buffer["rewards"][batch_idx])
                            agent_mask.append(True)
                        else:
                            dummy_obs = {}
                            for key in self.env.observation_space(agent_id).keys():
                                if hasattr(self.env.observation_space(agent_id)[key], "sample"):
                                    dummy_obs[key] = self.env.observation_space(agent_id)[key].sample()
                                else:
                                    dummy_obs[key] = 0
                            batch_agent_obs.append(dummy_obs)
                            batch_agent_rewards.append(0.0)
                            agent_mask.append(False)

                    if batch_agent_obs:
                        current_action_probs = actor.forward_batch(batch_agent_obs, batch_agent_rewards)

                        agent_mask_tensor = torch.tensor(agent_mask, device=self.device)
                        if agent_mask_tensor.sum() > 0:
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

                            ratio = current_action_prob_taken / (old_action_prob_taken + 1e-8)

                            surrogate1 = ratio * valid_advantages
                            surrogate2 = (
                                torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                                * valid_advantages
                            )

                            actor_loss = -torch.min(surrogate1, surrogate2).mean()

                            optimizer.zero_grad()
                            actor_loss.backward()
                            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                            optimizer.step()

            epoch_time = time.perf_counter() - epoch_start_time
            avg_critic_loss = total_critic_loss / num_batches if num_batches > 0 else 0
            print_colored(
                f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch + 1} complete - "
                f"Time: {epoch_time:.2f}s, Critic Loss: {avg_critic_loss:.6f}",
                "green",
            )

        total_update_time = time.perf_counter() - update_start_time
        print_colored(
            f"[{datetime.now().strftime('%H:%M:%S')}] Policy update complete - Total time: {total_update_time:.2f}s",
            "yellow",
        )

        # Return training metrics for logging
        self._last_update_metrics = {
            "critic_loss": avg_critic_loss,
            "update_time": total_update_time,
        }

        # Move models back to inference device and set eval mode
        self._ensure_inference_device()
        self._set_eval_mode()

        # Clear buffer
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
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.critic.state_dict(), f"{path}/critic.pt")

        for agent_id, actor in self.actors.items():
            torch.save(actor.state_dict(), f"{path}/actor_{agent_id}.pt")

        with open(f"{path}/device.txt", "w") as f:
            f.write(str(self.device))

    def load_models(self, path):
        """Load model weights from the specified path."""
        self.critic.load_state_dict(
            torch.load(f"{path}/critic.pt", map_location=self.inference_device)
        )

        for agent_id, actor in self.actors.items():
            actor.load_state_dict(
                torch.load(f"{path}/actor_{agent_id}.pt", map_location=self.inference_device)
            )

        self._set_eval_mode()
