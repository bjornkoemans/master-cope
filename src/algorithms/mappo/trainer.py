import numpy as np
import os
import time
import torch
from datetime import datetime

from ...core.display import print_colored

from .agent import MAPPOAgent


def get_model_architecture_summary(agent) -> str:
    """Get a detailed summary of the model architecture."""
    summary = []
    summary.append("Model Architecture")
    summary.append("=" * 18)

    # Get basic configuration
    summary.append(f"Number of agents: {agent.n_agents}")
    summary.append(f"Device: {agent.device}")
    summary.append(f"Gamma (discount factor): {agent.gamma}")
    summary.append(f"GAE Lambda: {agent.gae_lambda}")
    summary.append(f"Clip parameter: {agent.clip_param}")
    summary.append(f"Batch size: {agent.batch_size}")
    summary.append(f"Number of epochs: {agent.num_epochs}")
    summary.append("")

    # Actor networks
    summary.append("Actor Networks:")
    first_actor = next(iter(agent.actors.values()))
    total_actor_params = 0

    for agent_id, actor in agent.actors.items():
        actor_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
        total_actor_params += actor_params
        summary.append(f"  Agent {agent_id}: {actor_params:,} trainable parameters")

    summary.append(f"  Total actor parameters: {total_actor_params:,}")
    summary.append("")

    # Actor architecture details
    summary.append("Actor Architecture:")
    actor_layers = []
    for name, module in first_actor.named_modules():
        if isinstance(module, torch.nn.Linear):
            actor_layers.append(
                f"  {name}: Linear({module.in_features} -> {module.out_features})"
            )
        elif isinstance(module, torch.nn.Dropout):
            actor_layers.append(f"  {name}: Dropout(p={module.p})")

    if actor_layers:
        summary.extend(actor_layers)
    summary.append("")

    # Critic network
    summary.append("Critic Network:")
    critic_params = sum(p.numel() for p in agent.critic.parameters() if p.requires_grad)
    summary.append(f"  Trainable parameters: {critic_params:,}")
    summary.append("")

    # Critic architecture details
    summary.append("Critic Architecture:")
    critic_layers = []
    for name, module in agent.critic.named_modules():
        if isinstance(module, torch.nn.Linear):
            critic_layers.append(
                f"  {name}: Linear({module.in_features} -> {module.out_features})"
            )
        elif isinstance(module, torch.nn.Dropout):
            critic_layers.append(f"  {name}: Dropout(p={module.p})")

    if critic_layers:
        summary.extend(critic_layers)
    summary.append("")

    # Total parameters
    total_params = total_actor_params + critic_params
    summary.append(f"Total trainable parameters: {total_params:,}")
    summary.append("")

    # Optimizer information
    summary.append("Optimizers:")
    first_actor_optimizer = next(iter(agent.actor_optimizers.values()))
    summary.append(
        f"  Actor learning rate: {first_actor_optimizer.param_groups[0]['lr']}"
    )
    summary.append(
        f"  Critic learning rate: {agent.critic_optimizer.param_groups[0]['lr']}"
    )
    summary.append("")

    return "\n".join(summary)


def map_actions_to_array(actions: dict[int, int]) -> np.ndarray:
    """Maps the actions output dict to an array with the nth key mapped to the nth index."""
    array = np.zeros(len(actions))

    for i, key in enumerate(actions.keys()):
        array[i] = np.array(actions[key])

    return array


def map_action_probs_to_array(
    action_probs: dict[int, np.ndarray],
) -> np.ndarray:
    """Maps the action probabilities output dict to an array with the nth key mapped to the nth index. This is a 3D matrix"""
    array = np.zeros((len(action_probs)))

    for i, key in enumerate(action_probs.keys()):
        array[i] = action_probs[key][1]

    return array


class MAPPOTrainer:
    def __init__(
        self,
        env,
        mappo_agent: MAPPOAgent,
        total_training_episodes=50,  # Renamed for clarity
        eval_freq_episodes=1,
        save_freq_episodes=1,
        log_freq_episodes=10,
        eval_episodes=1,
        should_eval=True,
        experiment_dir="./experiments/mappo_default",
    ):
        self.env = env
        self.agent = mappo_agent
        self.total_training_episodes = total_training_episodes  # Renamed for clarity
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
        self.episodes_done = 0  # Renamed from epochs_done for clarity
        self.timesteps_done = 0
        self.best_eval_reward = -float("inf")
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.eval_rewards: list[float] = []

        # Initialize cumulative reward tracking
        self.cumulative_rewards: list[float] = []
        self.cumulative_eval_rewards: list[float] = []
        self.total_cumulative_reward = 0.0

    def train(self) -> list[float]:
        """Main training loop for MAPPO."""
        print(f"Starting MAPPO training for {self.total_training_episodes} episodes...")

        start_time = time.perf_counter()

        while self.episodes_done < self.total_training_episodes:
            # Create episode directory
            episode_dir = os.path.join(
                self.episodes_dir, f"episode_{self.episodes_done}"
            )
            os.makedirs(episode_dir, exist_ok=True)

            # Run one epoch (one complete episode)
            obs, _ = self.env.reset()
            episode_reward = 0.0  # Initialize as float
            episode_length = 0
            episode_time = time.perf_counter()

            # Arrays for storing episode data
            episode_actions: list[np.ndarray] = []
            episode_action_probs: list[np.ndarray] = []
            episode_rewards: list[float] = []
            episode_cumulative_rewards: list[float] = []
            episode_assigned_agents: list[int | None] = []  # Track assigned agents

            done = False
            while not done:
                # Select actions using the current policy
                actions, action_probs = self.agent.select_actions(obs)
                episode_actions.append(map_actions_to_array(actions))
                episode_action_probs.append(map_action_probs_to_array(action_probs))

                # Get state value
                value = self.agent.compute_values(obs)

                # Take actions in the environment
                next_obs, rewards, dones, truncated, _ = self.env.step(actions)
                step_reward = float(sum(rewards.values()))
                episode_rewards.append(step_reward)

                # Track assigned agent if a task was assigned
                if (
                    self.env.upcoming_case is not None
                    and self.env.upcoming_case.current_task is not None
                ):
                    # Get the selected agent ID from the environment
                    selected_agent_id = None
                    for agent_id, action in actions.items():
                        if action == 1 and self.env.agents[agent_id].can_perform_task(
                            self.env.upcoming_case.current_task.id
                        ):
                            selected_agent_id = agent_id
                            break
                    episode_assigned_agents.append(selected_agent_id)
                else:
                    episode_assigned_agents.append(None)

                # Update cumulative rewards
                self.total_cumulative_reward += step_reward
                self.cumulative_rewards.append(self.total_cumulative_reward)
                episode_cumulative_rewards.append(self.total_cumulative_reward)

                # Store experience
                done = any(list(dones.values()) + list(truncated.values()))
                self.agent.store_experience(
                    obs, actions, action_probs, step_reward, done, value
                )

                # Update episode tracking
                episode_reward += step_reward
                episode_length += 1

                # Move to the next step
                obs = next_obs
                self.timesteps_done += 1

            # Episode completed - update policy
            self.agent.update_policy()

            # Log performance
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            # Save episode data with resource names and agent assignments
            # Create header with resource names
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

            # Save action probabilities with header
            np.savetxt(
                os.path.join(episode_dir, "action_probs.csv"),
                episode_action_probs,
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
            with open(os.path.join(episode_dir, "summary.txt"), "w") as f:
                f.write(f"Episode {self.episodes_done}\n")
                f.write(f"Total Reward: {episode_reward:.2f}\n")
                f.write(f"Episode Length: {episode_length}\n")
                f.write(f"Time: {time.perf_counter() - episode_time:.2f} seconds\n")
                f.write(f"Cumulative Reward: {self.total_cumulative_reward:.2f}\n")

            self.episodes_done += 1

            # Logging
            if self.episodes_done % self.log_freq_episodes == 0:
                episode_time = time.perf_counter() - episode_time
                avg_reward = np.mean(self.episode_rewards[-self.log_freq_episodes :])
                avg_length = np.mean(self.episode_lengths[-self.log_freq_episodes :])
                print(
                    f"Episode {self.episodes_done}/{self.total_training_episodes} | "
                    f"Episode Reward: {episode_reward:.2f} | "
                    f"Episode Length: {episode_length} | "
                    f"Avg. Reward: {avg_reward:.2f} | "
                    f"Avg. Length: {avg_length:.2f} | "
                    f"Time for episode: {episode_time:.2f} seconds"
                )

            # Periodic evaluation
            if self.should_eval and self.episodes_done % self.eval_freq_episodes == 0:
                eval_reward, eval_cumulative_rewards = self.evaluate()
                self.eval_rewards.append(eval_reward)  # type: ignore
                self.cumulative_eval_rewards.extend(eval_cumulative_rewards)
                print_colored(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Evaluation at episode {self.episodes_done}: {eval_reward:.2f}"
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
                    self.agent.save_models(os.path.join(self.experiment_dir, "best"))
                    print(f"New best model saved with reward: {eval_reward:.2f}")

            # Periodic saving (every few episodes)
            if self.episodes_done % self.save_freq_episodes == 0:
                self.agent.save_models(
                    os.path.join(
                        self.experiment_dir, f"checkpoint_{self.episodes_done}"
                    )
                )
                print_colored(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint saved at episode {self.episodes_done}"
                )

        # Save final model and training summary
        self.agent.save_models(os.path.join(self.experiment_dir, "final"))

        # Save training summary and cumulative rewards
        with open(os.path.join(self.experiment_dir, "training_summary.txt"), "w") as f:
            f.write(f"Training completed after {self.episodes_done} episodes\n")
            f.write(f"Total episodes: {self.episodes_done}\n")
            f.write(f"Total timesteps: {self.timesteps_done}\n")
            f.write(
                f"Total time: {(time.perf_counter() - start_time) / 60:.2f} minutes\n"
            )
            f.write(f"Best evaluation reward: {self.best_eval_reward:.2f}\n")
            f.write(f"Final cumulative reward: {self.total_cumulative_reward:.2f}\n")
            f.write("\n")

            # Add model architecture information
            model_architecture = get_model_architecture_summary(self.agent)
            f.write(model_architecture)

            f.write("\nEpisode Rewards:\n")
            for i, reward in enumerate(self.episode_rewards):
                f.write(f"Episode {i}: {reward:.2f}\n")
            f.write("\nEvaluation Rewards:\n")
            for i, reward in enumerate(self.eval_rewards):
                f.write(f"Eval {i}: {reward:.2f}\n")

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

        print(
            f"Training completed after {self.episodes_done} episodes ({self.episodes_done} episodes and {self.timesteps_done} timesteps)."
        )
        print(f"Total time: {(time.perf_counter() - start_time) / 60:.2f} minutes")

        return self.episode_rewards

    def evaluate(self, deterministic=True):
        """Evaluate the current policy."""
        eval_rewards = []
        eval_cumulative_rewards = []
        print_colored(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Evaluating MAPPO agent for {self.eval_episodes} episodes...",
            "green",
        )
        for _ in range(self.eval_episodes):
            print_colored(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting evaluation episode {_ + 1}/{self.eval_episodes}",
                "green",
            )
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_cumulative_reward = 0

            iteration = 0

            while not done:
                # Select actions deterministically for evaluation
                actions, _ = self.agent.select_actions(obs, deterministic=deterministic)
                next_obs, rewards, dones, truncated, _ = self.env.step(actions)

                step_reward = float(sum(rewards.values()))
                episode_reward += step_reward
                episode_cumulative_reward += step_reward
                eval_cumulative_rewards.append(episode_cumulative_reward)

                # Check if episode is done
                done = any(list(dones.values()) + list(truncated.values()))
                obs = next_obs
                iteration += 1
            eval_rewards.append(episode_reward)

        avg_reward = np.mean(eval_rewards)
        return avg_reward, eval_rewards
