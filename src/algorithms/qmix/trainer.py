import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os
import time
from datetime import datetime

from ...core.display import print_colored


class QMIXTrainer:
    def __init__(
        self,
        env,
        agent,
        total_training_episodes=100,
        batch_size=32,
        buffer_size=10000,
        target_update_interval=100,
        eval_freq_episodes=10,
        save_freq_episodes=25,
        log_freq_episodes=10,
        eval_episodes=3,
        should_eval=True,
        experiment_dir="./experiments/qmix_default",
        **kwargs,
    ):
        self.env = env
        self.agent = agent
        self.total_training_episodes = total_training_episodes
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.target_update_interval = target_update_interval

        # Logging and evaluation parameters
        self.eval_freq_episodes = eval_freq_episodes
        self.save_freq_episodes = save_freq_episodes
        self.log_freq_episodes = log_freq_episodes
        self.should_eval = should_eval
        self.eval_episodes = eval_episodes
        self.experiment_dir = experiment_dir

        # Create experiment directory structure
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.episodes_dir = os.path.join(self.experiment_dir, "episodes")
        os.makedirs(self.episodes_dir, exist_ok=True)

        # Initialize tracking variables
        self.episodes_done = 0
        self.timesteps_done = 0
        self.best_eval_reward = -float("inf")
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.eval_rewards: list[float] = []
        self.episode_losses: list[float] = []

        # Initialize cumulative reward tracking
        self.cumulative_rewards: list[float] = []
        self.cumulative_eval_rewards: list[float] = []
        self.total_cumulative_reward = 0.0

    def train(self):
        """Main training loop for QMIX with comprehensive logging."""
        print_colored(
            f"Starting QMIX training for {self.total_training_episodes} episodes...",
            "green",
        )

        start_time = time.perf_counter()

        while self.episodes_done < self.total_training_episodes:
            # Create episode directory
            episode_dir = os.path.join(
                self.episodes_dir, f"episode_{self.episodes_done}"
            )
            os.makedirs(episode_dir, exist_ok=True)

            # Run one episode
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            episode_time = time.perf_counter()
            episode_loss = 0.0
            num_learning_steps = 0

            # Arrays for storing episode data
            episode_actions: list[np.ndarray] = []
            episode_q_values: list[np.ndarray] = []
            episode_rewards: list[float] = []
            episode_cumulative_rewards: list[float] = []
            episode_assigned_agents: list[int | None] = []

            step_count = 0
            while not done:
                step_count += 1
                if step_count % 100 == 0:  # Log every 100 steps
                    print_colored(
                        f"Episode {self.episodes_done}, Step {step_count}, Buffer size: {len(self.buffer)}",
                        "yellow",
                    )

                # Select actions using the current policy
                actions, _ = self.agent.select_actions(obs)

                # Get Q-values for logging
                q_values = self.agent.get_q_values(obs)
                episode_actions.append(self._map_actions_to_array(actions))
                episode_q_values.append(q_values.detach().cpu().numpy())

                # Take actions in the environment
                next_obs, rewards, terminations, truncations, _ = self.env.step(actions)
                reward = float(sum(rewards.values()))
                done = any(list(terminations.values()) + list(truncations.values()))

                # Track assigned agent if a task was assigned
                assigned_agent_id = self._get_assigned_agent(actions)
                episode_assigned_agents.append(assigned_agent_id)

                # Store experience for replay buffer
                global_state = self.agent.get_global_state(obs)
                next_global_state = (
                    self.agent.get_global_state(next_obs) if not done else None
                )

                self.buffer.append(
                    (
                        obs,
                        actions,
                        reward,
                        next_obs,
                        done,
                        global_state,
                        next_global_state,
                    )
                )

                # Update cumulative rewards
                self.total_cumulative_reward += reward
                self.cumulative_rewards.append(self.total_cumulative_reward)
                episode_cumulative_rewards.append(self.total_cumulative_reward)

                # Store episode data
                episode_rewards.append(reward)
                episode_reward += reward
                episode_length += 1
                self.timesteps_done += 1

                # Learn from experience
                if len(self.buffer) >= self.batch_size:
                    loss = self.learn()
                    if loss is not None:
                        episode_loss += loss
                        num_learning_steps += 1

                obs = next_obs

            # Update target networks
            if self.episodes_done % self.target_update_interval == 0:
                self.agent.update_target()
                print_colored(
                    f"Target networks updated at episode {self.episodes_done}", "cyan"
                )

            # Calculate average episode loss
            avg_episode_loss = episode_loss / max(1, num_learning_steps)

            # Store episode metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_losses.append(avg_episode_loss)

            # Save episode data with resource names and agent assignments
            resource_names = [agent.name for agent in self.env.agents]
            header = ";".join(resource_names)

            # Save actions with header
            np.savetxt(
                os.path.join(episode_dir, "actions.csv"),
                episode_actions,
                delimiter=";",
                header=header,
                comments="",
            )

            # Save Q-values with header
            # Convert list of arrays to 2D array for saving
            if episode_q_values:
                # Flatten each q_values array and stack them
                q_vals_flattened = []
                for q_vals in episode_q_values:
                    # Flatten the array to 1D
                    q_vals_flattened.append(q_vals.flatten())
                q_vals_array = np.array(q_vals_flattened)
            else:
                q_vals_array = np.array([])

            np.savetxt(
                os.path.join(episode_dir, "q_values.csv"),
                q_vals_array,
                delimiter=";",
                header=header,
                comments="",
            )

            # Save assigned agents
            np.savetxt(
                os.path.join(episode_dir, "assigned_agents.csv"),
                np.array(
                    episode_assigned_agents, dtype=float
                ),  # Convert to float array, None becomes nan
                delimiter=";",
                header="assigned_agent",
                comments="",
            )

            # Save other episode data
            np.savetxt(
                os.path.join(episode_dir, "rewards.csv"),
                episode_rewards,
                delimiter=";",
            )
            np.savetxt(
                os.path.join(episode_dir, "cumulative_rewards.csv"),
                episode_cumulative_rewards,
                delimiter=";",
            )

            # Save episode summary
            episode_time_elapsed = time.perf_counter() - episode_time
            with open(os.path.join(episode_dir, "summary.txt"), "w") as f:
                f.write(f"Episode {self.episodes_done}\n")
                f.write(f"Total Reward: {episode_reward:.2f}\n")
                f.write(f"Episode Length: {episode_length}\n")
                f.write(f"Average Loss: {avg_episode_loss:.6f}\n")
                f.write(f"Epsilon: {self.agent.epsilon:.4f}\n")
                f.write(f"Buffer Size: {len(self.buffer)}\n")
                f.write(f"Learning Steps: {num_learning_steps}\n")
                f.write(f"Time: {episode_time_elapsed:.2f} seconds\n")
                f.write(f"Cumulative Reward: {self.total_cumulative_reward:.2f}\n")

            self.episodes_done += 1

            # Logging
            if self.episodes_done % self.log_freq_episodes == 0:
                avg_reward = np.mean(self.episode_rewards[-self.log_freq_episodes :])
                avg_length = np.mean(self.episode_lengths[-self.log_freq_episodes :])
                avg_loss = np.mean(self.episode_losses[-self.log_freq_episodes :])
                print_colored(
                    f"Episode {self.episodes_done}/{self.total_training_episodes} | "
                    f"Episode Reward: {episode_reward:.2f} | "
                    f"Episode Length: {episode_length} | "
                    f"Avg. Reward: {avg_reward:.2f} | "
                    f"Avg. Length: {avg_length:.2f} | "
                    f"Avg. Loss: {avg_loss:.6f} | "
                    f"Epsilon: {self.agent.epsilon:.4f} | "
                    f"Buffer: {len(self.buffer)} | "
                    f"Time: {episode_time_elapsed:.2f}s",
                    "blue",
                )

            # Periodic evaluation
            if self.should_eval and self.episodes_done % self.eval_freq_episodes == 0:
                eval_reward, eval_cumulative_rewards = self.evaluate()
                self.eval_rewards.append(float(eval_reward))
                self.cumulative_eval_rewards.extend(eval_cumulative_rewards)
                print_colored(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Evaluation at episode {self.episodes_done}: {eval_reward:.2f}",
                    "green",
                )

                # Save evaluation results
                eval_dir = os.path.join(episode_dir, "evaluation")
                os.makedirs(eval_dir, exist_ok=True)
                np.savetxt(
                    os.path.join(eval_dir, "eval_reward.csv"),
                    [eval_reward],
                    delimiter=";",
                )
                np.savetxt(
                    os.path.join(eval_dir, "eval_cumulative_rewards.csv"),
                    eval_cumulative_rewards,
                    delimiter=";",
                )

                # Save best model
                if eval_reward > self.best_eval_reward:
                    self.best_eval_reward = eval_reward
                    self.agent.save_models(
                        os.path.join(self.experiment_dir, "best.pth")
                    )
                    print_colored(
                        f"New best model saved with reward: {eval_reward:.2f}", "green"
                    )

            # Periodic saving (every few episodes)
            if self.episodes_done % self.save_freq_episodes == 0:
                checkpoint_path = os.path.join(
                    self.experiment_dir, f"checkpoint_{self.episodes_done}.pth"
                )
                self.agent.save_models(checkpoint_path)
                print_colored(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint saved at episode {self.episodes_done}",
                    "cyan",
                )

        # Save final model and training summary
        self.agent.save_models(os.path.join(self.experiment_dir, "final.pth"))

        # Save training summary and cumulative rewards
        self._save_training_summary(start_time)

        # Save cumulative rewards for plotting
        np.savetxt(
            os.path.join(self.experiment_dir, "cumulative_rewards.csv"),
            self.cumulative_rewards,
            delimiter=";",
        )
        np.savetxt(
            os.path.join(self.experiment_dir, "cumulative_eval_rewards.csv"),
            self.cumulative_eval_rewards,
            delimiter=";",
        )

        total_time = (time.perf_counter() - start_time) / 60
        print_colored(
            f"Training completed after {self.episodes_done} episodes ({self.timesteps_done} timesteps).",
            "green",
        )
        print_colored(f"Total time: {total_time:.2f} minutes", "green")

        return self.episode_rewards

    def learn(self):
        """Learn from a batch of experiences and return the loss."""
        if len(self.buffer) < self.batch_size:
            return None

        # Sample a batch from the replay buffer
        batch_indices = random.sample(range(len(self.buffer)), self.batch_size)
        batch = [self.buffer[idx] for idx in batch_indices]

        (
            obs_batch,
            actions_batch,
            reward_batch,
            next_obs_batch,
            done_batch,
            state_batch,
            next_state_batch,
        ) = zip(*batch)

        # Convert to tensors
        reward_batch = torch.FloatTensor(reward_batch).to(self.agent.device)
        done_batch = torch.BoolTensor(done_batch).to(self.agent.device)

        # Get current Q-values for all agents
        current_q_values_list = []
        for obs in obs_batch:
            # Get Q-values for all agents
            obs_arrays = []
            for agent in self.env.agents:
                flat_obs = self.agent._flatten_observation(obs[agent.id])
                obs_arrays.append(flat_obs)
            obs_tensor = torch.tensor(
                np.stack(obs_arrays), dtype=torch.float32, device=self.agent.device
            )
            q_vals = self.agent.agent_net(obs_tensor)
            current_q_values_list.append(q_vals)
        current_q_values = torch.stack(
            current_q_values_list
        )  # [batch_size, n_agents, n_actions]

        # Get Q-values for chosen actions
        chosen_actions = []
        for actions in actions_batch:
            action_tensor = torch.LongTensor(
                [actions[agent.id] for agent in self.env.agents]
            )
            chosen_actions.append(action_tensor)
        chosen_actions = torch.stack(chosen_actions).to(
            self.agent.device
        )  # [batch_size, n_agents]

        chosen_q_values = torch.gather(
            current_q_values, dim=2, index=chosen_actions.unsqueeze(2)
        ).squeeze(2)

        # Get next Q-values for target network
        target_q_values_list = []
        for i, next_obs in enumerate(next_obs_batch):
            if done_batch[i]:
                # If episode is done, Q-value is 0
                target_q_vals = torch.zeros(
                    len(self.env.agents), self.agent.n_actions
                ).to(self.agent.device)
            else:
                with torch.no_grad():
                    obs_arrays = []
                    for agent in self.env.agents:
                        flat_obs = self.agent._flatten_observation(next_obs[agent.id])
                        obs_arrays.append(flat_obs)
                    obs_tensor = torch.tensor(
                        np.stack(obs_arrays),
                        dtype=torch.float32,
                        device=self.agent.device,
                    )
                    target_q_vals = self.agent.target_agent_net(obs_tensor)
            target_q_values_list.append(target_q_vals)

        target_q_values = torch.stack(
            target_q_values_list
        )  # [batch_size, n_agents, n_actions]
        target_max_q_values = target_q_values.max(dim=2)[0]  # [batch_size, n_agents]

        # Convert states to tensors
        state_tensors = torch.FloatTensor(np.array(state_batch)).to(self.agent.device)
        next_state_tensors = []
        for i, next_state in enumerate(next_state_batch):
            if done_batch[i]:
                next_state_tensors.append(torch.zeros_like(state_tensors[0]))
            else:
                next_state_tensors.append(
                    torch.FloatTensor(next_state).to(self.agent.device)
                )
        next_state_tensors = torch.stack(next_state_tensors)

        # Calculate mixed Q-values
        current_mixed_q = self.agent.mixing_net(chosen_q_values, state_tensors)

        with torch.no_grad():
            target_mixed_q = self.agent.target_mixing_net(
                target_max_q_values, next_state_tensors
            )
            target_q = reward_batch.unsqueeze(1) + self.agent.gamma * target_mixed_q * (
                ~done_batch
            ).unsqueeze(1)

        # Calculate loss
        loss = F.mse_loss(current_mixed_q, target_q)

        # Optimize
        self.agent.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.agent.agent_net.parameters())
            + list(self.agent.mixing_net.parameters()),
            10.0,
        )
        self.agent.optimizer.step()

        return loss.item()

    def _map_actions_to_array(self, actions: dict[int, int]) -> np.ndarray:
        """Maps the actions output dict to an array with the nth key mapped to the nth index."""
        array = np.zeros(len(actions))
        for i, agent in enumerate(self.env.agents):
            array[i] = actions[agent.id]
        return array

    def _get_assigned_agent(self, actions: dict[int, int]) -> int | None:
        """Get the ID of the agent that was assigned a task, if any."""
        if (
            self.env.upcoming_case is not None
            and self.env.upcoming_case.current_task is not None
        ):
            # Get the selected agent ID from the environment
            for agent_id, action in actions.items():
                if action == 1 and self.env.agents[agent_id].can_perform_task(
                    self.env.upcoming_case.current_task.id
                ):
                    return agent_id
        return None

    def evaluate(self, deterministic=True):
        """Evaluate the current policy."""
        eval_rewards = []
        eval_cumulative_rewards = []
        print_colored(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Evaluating QMIX agent for {self.eval_episodes} episodes...",
            "green",
        )

        for ep in range(self.eval_episodes):
            print_colored(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting evaluation episode {ep + 1}/{self.eval_episodes}",
                "green",
            )
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_cumulative_reward = 0

            while not done:
                # Select actions deterministically for evaluation
                actions, _ = self.agent.select_actions(obs, deterministic=deterministic)
                next_obs, rewards, terminations, truncations, _ = self.env.step(actions)

                step_reward = float(sum(rewards.values()))
                episode_reward += step_reward
                episode_cumulative_reward += step_reward
                eval_cumulative_rewards.append(episode_cumulative_reward)

                # Check if episode is done
                done = any(list(terminations.values()) + list(truncations.values()))
                obs = next_obs

            eval_rewards.append(episode_reward)

        avg_reward = np.mean(eval_rewards)
        return avg_reward, eval_cumulative_rewards

    def _save_training_summary(self, start_time):
        """Save a comprehensive training summary."""
        with open(os.path.join(self.experiment_dir, "training_summary.txt"), "w") as f:
            f.write("QMIX Training Summary\n")
            f.write("=" * 25 + "\n")
            f.write(f"Training episodes: {self.episodes_done}\n")
            f.write(f"Total timesteps: {self.timesteps_done}\n")
            f.write(f"Buffer size: {len(self.buffer)}\n")
            f.write(f"Target update interval: {self.target_update_interval}\n")
            f.write(f"Batch size: {self.batch_size}\n")
            f.write(f"Final epsilon: {self.agent.epsilon:.4f}\n")
            f.write(
                f"Total time: {(time.perf_counter() - start_time) / 60:.2f} minutes\n"
            )
            f.write(f"Best evaluation reward: {self.best_eval_reward:.2f}\n")
            f.write(f"Final cumulative reward: {self.total_cumulative_reward:.2f}\n")
            f.write("\n")

            # Add model architecture information
            model_architecture = self._get_model_architecture_summary()
            f.write(model_architecture)

            f.write("\nEpisode Rewards:\n")
            f.write("=" * 16 + "\n")
            for i, reward in enumerate(self.episode_rewards):
                f.write(f"Episode {i+1:3d}: {reward:8.2f}\n")

            if self.eval_rewards:
                f.write("\nEvaluation Rewards:\n")
                f.write("=" * 19 + "\n")
                for i, reward in enumerate(self.eval_rewards):
                    f.write(f"Eval {i+1:3d}: {reward:8.2f}\n")

            # Add summary statistics
            if self.episode_rewards:
                f.write("\nTraining Reward Statistics:\n")
                f.write("=" * 28 + "\n")
                f.write(f"  Total episodes: {len(self.episode_rewards)}\n")
                f.write(f"  Average reward: {np.mean(self.episode_rewards):8.2f}\n")
                f.write(f"  Best reward:    {np.max(self.episode_rewards):8.2f}\n")
                f.write(f"  Worst reward:   {np.min(self.episode_rewards):8.2f}\n")
                f.write(f"  Std deviation:  {np.std(self.episode_rewards):8.2f}\n")

            if self.episode_losses:
                f.write("\nTraining Loss Statistics:\n")
                f.write("=" * 26 + "\n")
                f.write(f"  Average loss:   {np.mean(self.episode_losses):8.6f}\n")
                f.write(f"  Final loss:     {self.episode_losses[-1]:8.6f}\n")
                f.write(f"  Std deviation:  {np.std(self.episode_losses):8.6f}\n")

    def _get_model_architecture_summary(self) -> str:
        """Get a detailed summary of the QMIX model architecture."""
        summary = []
        summary.append("\nQMIX Model Architecture")
        summary.append("=" * 24)

        # Get basic configuration
        summary.append(f"Number of agents: {self.agent.n_agents}")
        summary.append(f"Observation dimension: {self.agent.obs_dim}")
        summary.append(f"State dimension: {self.agent.state_dim}")
        summary.append(f"Number of actions: {self.agent.n_actions}")
        summary.append(f"Device: {self.agent.device}")
        summary.append(f"Gamma (discount factor): {self.agent.gamma}")
        summary.append(f"Final epsilon: {self.agent.epsilon:.4f}")
        summary.append("")

        # Agent network
        summary.append("Agent Network:")
        agent_params = sum(
            p.numel() for p in self.agent.agent_net.parameters() if p.requires_grad
        )
        summary.append(f"  Trainable parameters: {agent_params:,}")

        # Agent network architecture details
        summary.append("  Architecture:")
        for name, module in self.agent.agent_net.named_modules():
            if isinstance(module, torch.nn.Linear):
                summary.append(
                    f"    {name}: Linear({module.in_features} -> {module.out_features})"
                )

        summary.append("")

        # Mixing network
        summary.append("Mixing Network:")
        mixing_params = sum(
            p.numel() for p in self.agent.mixing_net.parameters() if p.requires_grad
        )
        summary.append(f"  Trainable parameters: {mixing_params:,}")

        # Mixing network architecture details
        summary.append("  Architecture:")
        for name, module in self.agent.mixing_net.named_modules():
            if isinstance(module, torch.nn.Linear):
                summary.append(
                    f"    {name}: Linear({module.in_features} -> {module.out_features})"
                )

        summary.append("")

        # Total parameters
        total_params = agent_params + mixing_params
        summary.append(f"Total trainable parameters: {total_params:,}")
        summary.append("")

        # Optimizer information
        summary.append("Optimizer:")
        summary.append(f"  Learning rate: {self.agent.optimizer.param_groups[0]['lr']}")
        summary.append("")

        return "\n".join(summary)
