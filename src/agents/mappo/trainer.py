import math
import numpy as np
import os
import time
import torch
import pandas as pd
from datetime import datetime

from environment.display import print_colored

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
    summary.append(f"Buffer size: {agent.buffer_size or 'unlimited (full episode)'}")
    summary.append(f"COMA (counterfactual baseline): {'enabled' if agent.use_coma else 'DISABLED (shared GAE)'}")
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
        test_data=None,
        fitted_distributions=None,
        entropy_coef=0.05,
        entropy_coef_min=0.03,
        warmup_frac=0.3,
        skip_idle_steps=False,
        n_final_eval_episodes=10,
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
        self.test_data = test_data
        self.fitted_distributions = fitted_distributions
        self.entropy_coef_start = entropy_coef
        self.entropy_coef_min = entropy_coef_min
        self.warmup_frac = warmup_frac
        self.skip_idle_steps = skip_idle_steps
        self.n_final_eval_episodes = n_final_eval_episodes

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

    def save_trainer_state(self) -> None:
        """Save trainer state for resume (episodes_done, rewards, etc.)."""
        import json
        state = {
            "episodes_done": self.episodes_done,
            "timesteps_done": self.timesteps_done,
            "best_eval_reward": self.best_eval_reward,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "eval_rewards": self.eval_rewards,
            "cumulative_rewards": self.cumulative_rewards,
            "cumulative_eval_rewards": self.cumulative_eval_rewards,
            "total_cumulative_reward": self.total_cumulative_reward,
        }
        state_path = os.path.join(self.experiment_dir, "trainer_state.json")
        with open(state_path, "w") as f:
            json.dump(state, f)

    def load_trainer_state(self) -> bool:
        """Load trainer state from a previous run. Returns True if successful."""
        import json
        state_path = os.path.join(self.experiment_dir, "trainer_state.json")
        if not os.path.exists(state_path):
            return False
        with open(state_path, "r") as f:
            state = json.load(f)
        self.episodes_done = state["episodes_done"]
        self.timesteps_done = state["timesteps_done"]
        self.best_eval_reward = state["best_eval_reward"]
        self.episode_rewards = state["episode_rewards"]
        self.episode_lengths = state["episode_lengths"]
        self.eval_rewards = state["eval_rewards"]
        self.cumulative_rewards = state["cumulative_rewards"]
        self.cumulative_eval_rewards = state["cumulative_eval_rewards"]
        self.total_cumulative_reward = state["total_cumulative_reward"]
        return True

    def resume(self, additional_episodes: int) -> list[float]:
        """Resume training from last checkpoint for additional episodes."""
        checkpoint_path = os.path.join(self.experiment_dir, "checkpoints", "final")
        if not os.path.exists(checkpoint_path):
            print("No final checkpoint found. Cannot resume.")
            return []

        # Load model weights + optimizer states
        self.agent.load_models(checkpoint_path, load_optimizers=True)
        print_colored(f"Loaded model + optimizer states from {checkpoint_path}", "green")

        # Load trainer state
        if not self.load_trainer_state():
            print("No trainer state found. Cannot resume.")
            return []

        prev_episodes = self.episodes_done
        self.total_training_episodes = prev_episodes + additional_episodes
        print_colored(
            f"Resuming from episode {prev_episodes}, training {additional_episodes} more "
            f"(total: {self.total_training_episodes})",
            "green",
        )

        # Continue training loop
        return self.train()

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

            # Print episode header
            print(f"\n{'='*70}")
            print(f"EPISODE {self.episodes_done + 1}/{self.total_training_episodes} - COLLECTING ROLLOUT DATA")
            print(f"{'='*70}")

            # Run one epoch (one complete episode)
            obs, _ = self.env.reset(options={"phase": f"train_ep{self.episodes_done}"})
            self.agent.set_current_episode(self.episodes_done)  # Update comm warmup counter
            self.agent.reset_history()  # Reset LSTM hidden states for IC3Net
            episode_reward = 0.0  # Initialize as float
            episode_length = 0
            episode_time = time.perf_counter()

            # Arrays for storing episode data
            episode_actions: list[np.ndarray] = []
            episode_action_probs: list[np.ndarray] = []
            episode_rewards: list[float] = []
            episode_cumulative_rewards: list[float] = []
            episode_assigned_agents: list[int | None] = []  # Track assigned agents
            episode_assigned_agent_names: list[str] = []  # Track assigned agent names
            episode_activity_names: list[str] = []  # Track activity names
            # Volunteer rate tracking
            volunteer_decisions = 0  # steps where at least 1 agent volunteered
            total_decisions = 0  # total steps where a decision was made
            # Per-agent p(volunteer) tracking
            agent_vol_probs: dict[int, list[float]] = {agent.id: [] for agent in self.env.agents}
            # Gate activation tracking (IC3Net only)
            gate_open_count = 0
            gate_total_count = 0
            # Per-timestep gate log: list of dicts for detailed analysis
            gate_log: list[dict] = []

            done = False
            while not done:
                # Select actions using the current policy
                actions, action_probs = self.agent.select_actions(obs)
                episode_actions.append(map_actions_to_array(actions))
                episode_action_probs.append(map_action_probs_to_array(action_probs))

                # Get state value
                value = self.agent.compute_values(obs)

                # Track assigned task BEFORE step() (upcoming_case is consumed by step)
                if (
                    self.env.upcoming_case is not None
                    and self.env.upcoming_case.current_task is not None
                ):
                    task_id = self.env.upcoming_case.current_task.id
                    activity_name = self.env.inv_task_dict.get(task_id, str(task_id))
                    step_has_task = True
                else:
                    activity_name = ""
                    step_has_task = False

                # Take actions in the environment
                next_obs, rewards, dones, truncated, _ = self.env.step(actions)
                # rewards is dict[agent_id, float] — may be per-agent or shared
                # For logging, use mean across agents
                step_reward_mean = float(np.mean(list(rewards.values())))
                episode_rewards.append(step_reward_mean)
                # Pass full per-agent rewards dict to agent for individual GAE
                step_rewards = {aid: float(r) for aid, r in rewards.items()}

                # Track volunteer rate: count if any capable agent volunteered
                any_volunteered = any(action == 1 for action in actions.values())
                if any_volunteered or step_reward_mean != 0.0:
                    # Only count steps where a decision was relevant
                    total_decisions += 1
                    if any_volunteered:
                        volunteer_decisions += 1

                # Track per-agent p(volunteer) for monitoring learning progress
                for agent_id, probs in action_probs.items():
                    if hasattr(probs, '__len__') and len(probs) > 1:
                        p_vol = float(probs[1]) if not isinstance(probs[1], float) else probs[1]
                        agent_vol_probs[agent_id].append(p_vol)

                # Track gate activation (IC3Net / MLP comm only)
                if hasattr(self.agent, 'ic3net_actor') and self.agent.ic3net_actor is not None:
                    gate_actions = getattr(self.agent.ic3net_actor, '_last_gate_actions', None)
                    if gate_actions is not None:
                        gate_open_count += int(gate_actions.sum().item())
                        gate_total_count += int(gate_actions.numel())
                        # Per-timestep gate log: agent-level gate decisions
                        # gate_actions shape: (n_agents, n_comm_rounds)
                        # Sum over comm rounds: >0 means gate was open at least once
                        gate_per_agent = (gate_actions.sum(dim=-1) > 0).int().tolist()
                        gate_entry = {
                            "step": episode_length,
                            "activity": activity_name,
                        }
                        for i, agent in enumerate(sorted(self.env.agents, key=lambda a: a.id)):
                            gate_entry[agent.name] = gate_per_agent[i]
                        gate_log.append(gate_entry)

                # Log assigned agent (captured before step)
                if step_has_task:
                    # Find which agent was actually assigned by checking actions
                    selected_agent_id = None
                    for agent_id, action in actions.items():
                        if action == 1 and self.env.agents[agent_id].can_perform_task(task_id):
                            selected_agent_id = agent_id
                            break
                    episode_assigned_agents.append(selected_agent_id)
                    episode_assigned_agent_names.append(
                        self.env.agents[selected_agent_id].name if selected_agent_id is not None else ""
                    )
                    episode_activity_names.append(activity_name)
                else:
                    episode_assigned_agents.append(None)
                    episode_assigned_agent_names.append("")
                    episode_activity_names.append("")

                # Update cumulative rewards
                self.total_cumulative_reward += step_reward_mean
                self.cumulative_rewards.append(self.total_cumulative_reward)
                episode_cumulative_rewards.append(self.total_cumulative_reward)

                # Store experience (pass per-agent rewards dict)
                done = any(list(dones.values()) + list(truncated.values()))

                # Skip idle steps: don't store experiences where no task was offered
                # This reduces noise in the training buffer (actions had no effect)
                # step_has_task is pre-computed at line 288-297
                should_store = True
                if self.skip_idle_steps and not step_has_task and step_reward_mean == 0.0 and not done:
                    should_store = False

                if should_store:
                    self.agent.store_experience(
                        obs, actions, action_probs, step_rewards, done, value
                    )

                # Update episode tracking
                episode_reward += step_reward_mean
                episode_length += 1

                # Move to the next step
                obs = next_obs
                self.timesteps_done += 1

                # Print progress every 5000 steps
                if episode_length % 5000 == 0:
                    elapsed = time.perf_counter() - episode_time
                    print(f"  Step {episode_length:>6} | Reward: {episode_reward:>12.2f} | Time: {elapsed:>6.1f}s")

            # Episode completed
            episode_elapsed = time.perf_counter() - episode_time
            print(f"\nRollout complete: {episode_length} steps in {episode_elapsed:.1f}s")
            print(f"Episode reward: {episode_reward:.2f}")

            # Update policy with warmup + cosine entropy decay
            # Phase 1 (warmup): keep entropy constant for first warmup_frac of training
            # Phase 2 (decay): cosine decay from entropy_coef_start to entropy_coef_min
            progress = self.episodes_done / self.total_training_episodes
            if progress < self.warmup_frac:
                entropy_coef = self.entropy_coef_start  # Constant during warmup
            else:
                decay_progress = (progress - self.warmup_frac) / (1.0 - self.warmup_frac)
                entropy_coef = self.entropy_coef_min + 0.5 * (
                    self.entropy_coef_start - self.entropy_coef_min
                ) * (1.0 + math.cos(math.pi * decay_progress))
            print(f"\n{'='*70}")
            print(f"POLICY UPDATE - TRAINING NEURAL NETWORKS (entropy_coef={entropy_coef:.4f})")
            print(f"{'='*70}")
            self.agent.update_policy(entropy_coef=entropy_coef)

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

            # Save assigned agents with names and activity names (like old system)
            assigned_agents_df = pd.DataFrame({
                "assigned_agent_id": episode_assigned_agents,
                "assigned_agent_name": episode_assigned_agent_names,
                "activity_name": episode_activity_names,
            })
            assigned_agents_df.to_csv(
                os.path.join(episode_dir, "assigned_agents.csv"),
                index=False,
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

            # Compute volunteer rate
            vol_rate = (volunteer_decisions / total_decisions * 100) if total_decisions > 0 else 0.0
            print(f"  Volunteer rate: {vol_rate:.1f}% ({volunteer_decisions}/{total_decisions} decisions)")

            # Log per-agent average p(volunteer) — monitors learning even when p < 0.5
            avg_p_vol_per_agent = {}
            for agent_id, probs_list in agent_vol_probs.items():
                if probs_list:
                    avg_p = np.mean(probs_list)
                    avg_p_vol_per_agent[agent_id] = avg_p
            if avg_p_vol_per_agent:
                overall_avg_p = np.mean(list(avg_p_vol_per_agent.values()))
                agent_str = ", ".join(f"A{aid}={p:.3f}" for aid, p in sorted(avg_p_vol_per_agent.items()))
                print(f"  Avg p(volunteer): {overall_avg_p:.3f} [{agent_str}]")

            # Save episode summary
            with open(os.path.join(episode_dir, "summary.txt"), "w") as f:
                f.write(f"Episode {self.episodes_done}\n")
                f.write(f"Total Reward: {episode_reward:.2f}\n")
                f.write(f"Episode Length: {episode_length}\n")
                f.write(f"Time: {time.perf_counter() - episode_time:.2f} seconds\n")
                f.write(f"Cumulative Reward: {self.total_cumulative_reward:.2f}\n")
                f.write(f"Volunteer Rate: {vol_rate:.1f}% ({volunteer_decisions}/{total_decisions})\n")
                if avg_p_vol_per_agent:
                    f.write(f"Avg p(volunteer): {overall_avg_p:.3f}\n")
                    for aid, p in sorted(avg_p_vol_per_agent.items()):
                        f.write(f"  Agent {aid}: {p:.3f}\n")
                f.write(f"Entropy Coefficient: {entropy_coef:.6f}\n")
                # Loss metrics from last policy update
                update_metrics = getattr(self.agent, '_last_update_metrics', None)
                if update_metrics:
                    f.write(f"Critic Loss: {update_metrics.get('critic_loss', 0):.6f}\n")
                # Gate activation metrics (IC3Net / MLP comm only)
                if gate_total_count > 0:
                    gate_rate = gate_open_count / gate_total_count
                    f.write(f"Gate Activation Rate: {gate_rate:.4f} ({gate_open_count}/{gate_total_count})\n")

            # Save per-timestep gate activation log
            if gate_log:
                gate_df = pd.DataFrame(gate_log)
                gate_df.to_csv(
                    os.path.join(episode_dir, "gate_activations.csv"),
                    index=False,
                )

            self.episodes_done += 1

            # Logging
            if self.episodes_done % self.log_freq_episodes == 0:
                episode_time_total = time.perf_counter() - episode_time
                avg_reward = np.mean(self.episode_rewards[-self.log_freq_episodes :])
                avg_length = np.mean(self.episode_lengths[-self.log_freq_episodes :])
                print(f"\n{'='*70}")
                print(f"EPISODE {self.episodes_done}/{self.total_training_episodes} COMPLETE")
                print(f"{'='*70}")
                print(f"  Episode Reward:  {episode_reward:>12.2f}")
                print(f"  Episode Length:  {episode_length:>12}")
                print(f"  Avg Reward:      {avg_reward:>12.2f} (last {self.log_freq_episodes} episodes)")
                print(f"  Avg Length:      {avg_length:>12.2f} (last {self.log_freq_episodes} episodes)")
                print(f"  Total Time:      {episode_time_total:>12.2f}s")

            # Periodic evaluation
            if self.should_eval and self.episodes_done % self.eval_freq_episodes == 0:
                print(f"\n{'='*70}")
                print(f"EVALUATION - TESTING CURRENT POLICY")
                print(f"{'='*70}")
                eval_reward, eval_cumulative_rewards = self.evaluate(after_episode=self.episodes_done)
                self.eval_rewards.append(eval_reward)  # type: ignore
                self.cumulative_eval_rewards.extend(eval_cumulative_rewards)
                print(f"Evaluation reward: {eval_reward:.2f}")

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
                    self.agent.save_models(os.path.join(self.experiment_dir, "checkpoints", "best"))
                    print(f"New best model saved with reward: {eval_reward:.2f}")

            # Periodic saving (every few episodes)
            if self.episodes_done % self.save_freq_episodes == 0:
                self.agent.save_models(
                    os.path.join(
                        self.experiment_dir, "checkpoints", f"checkpoint_{self.episodes_done}"
                    )
                )
                print_colored(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint saved at episode {self.episodes_done}"
                )

        # Save final model, optimizer states, and trainer state for resume
        self.agent.save_models(os.path.join(self.experiment_dir, "checkpoints", "final"))
        self.save_trainer_state()

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

        # Run final evaluation on test data (like old system)
        if self.test_data is not None and self.fitted_distributions is not None:
            self.final_evaluation(n_episodes=self.n_final_eval_episodes)

        return self.episode_rewards

    def evaluate(self, deterministic=True, after_episode=None):
        """Evaluate the current policy."""
        eval_rewards = []
        eval_cumulative_rewards = []
        print_colored(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Evaluating MAPPO agent for {self.eval_episodes} episodes...",
            "green",
        )
        for eval_ep in range(self.eval_episodes):
            print_colored(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting evaluation episode {eval_ep + 1}/{self.eval_episodes}",
                "green",
            )
            ep_label = f"ep{after_episode}" if after_episode is not None else ""
            obs, _ = self.env.reset(options={"phase": f"eval_{ep_label}_e{eval_ep}"})
            self.agent.reset_history()  # Reset LSTM hidden states for IC3Net
            done = False
            episode_reward = 0
            episode_cumulative_reward = 0

            iteration = 0

            while not done:
                # Select actions deterministically for evaluation
                actions, _ = self.agent.select_actions(obs, deterministic=deterministic)
                next_obs, rewards, dones, truncated, _ = self.env.step(actions)

                step_reward = float(np.mean(list(rewards.values())))
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

    def final_evaluation(self, n_episodes: int = 10):
        """Run final evaluation on test data with a separate environment.

        Creates a new AgentOptimizerEnvironment using the test_data and the
        fitted distributions from training. Runs deterministic evaluation
        episodes and saves results to experiment_dir/final_evaluation/.
        """
        from environment.simulator import AgentOptimizerEnvironment
        import pandas as pd

        print(f"\n{'='*70}")
        print("FINAL EVALUATION ON TEST DATA")
        print(f"{'='*70}")

        # Load best model checkpoint (highest eval reward during training)
        best_model_path = os.path.join(self.experiment_dir, "checkpoints", "best")
        if os.path.exists(best_model_path):
            print(f"Loading best model from: {best_model_path} (eval reward: {self.best_eval_reward:.2f})")
            self.agent.load_models(best_model_path)
        else:
            print("Warning: No best model found, using last training checkpoint.")

        # Create separate evaluation environment with test data and pre-fitted distributions
        # Work schedule setting inherited from training env config
        eval_env = AgentOptimizerEnvironment(
            data=self.test_data,
            simulation_parameters={
                "start_timestamp": self.test_data["assign_timestamp"].min()
                if "assign_timestamp" in self.test_data.columns
                else self.test_data["start_timestamp"].min()
            },
            experiment_dir=self.experiment_dir,
            enable_logging=True,
            verbose=True,
            pre_fitted_distributions=self.fitted_distributions,
            max_steps=self.env.max_steps,
            max_episodes=self.env.max_episodes,
            work_schedule_enabled=self.env.work_schedule_enabled,
            work_start_hour=self.env.work_start_hour,
            work_end_hour=self.env.work_end_hour,
            reward_config=self.env.reward_config,
            parallel_task_groups=self.env._parallel_task_groups_config,
            use_agent_identity=self.env.use_agent_identity,
            fixed_agent_list=self.env.resources,
        )

        # Create output directory (rename previous one if resuming)
        final_eval_dir = os.path.join(self.experiment_dir, "final_evaluation")
        if os.path.exists(final_eval_dir):
            # Archive previous final_eval with episode count
            prev_state_file = os.path.join(self.experiment_dir, "trainer_state.json")
            prev_eps = "unknown"
            if os.path.exists(prev_state_file):
                import json
                try:
                    prev_state = json.load(open(prev_state_file))
                    prev_eps = prev_state.get("episodes_done", "unknown")
                except:
                    pass
            archive_name = f"final_evaluation_ep{prev_eps}"
            archive_path = os.path.join(self.experiment_dir, archive_name)
            if not os.path.exists(archive_path):
                os.rename(final_eval_dir, archive_path)
                print(f"  Archived previous final_evaluation → {archive_name}/")
            else:
                import shutil
                shutil.rmtree(final_eval_dir)
                print(f"  Removed previous final_evaluation (archive already exists)")
        os.makedirs(final_eval_dir, exist_ok=True)

        eval_rewards = []
        eval_lengths = []

        for ep in range(n_episodes):
            print_colored(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Final eval episode {ep + 1}/{n_episodes}",
                "green",
            )
            obs, _ = eval_env.reset(options={"phase": f"final_eval_ep{ep}"})
            self.agent.reset_history()  # Reset LSTM hidden states for IC3Net
            done = False
            episode_reward = 0.0
            episode_length = 0

            while not done:
                actions, _ = self.agent.select_actions(obs, deterministic=True)
                next_obs, rewards, dones, truncated, _ = eval_env.step(actions)

                step_reward = float(np.mean(list(rewards.values())))
                episode_reward += step_reward
                episode_length += 1

                done = any(list(dones.values()) + list(truncated.values()))
                obs = next_obs

            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            print(f"  Episode {ep + 1}: reward={episode_reward:.2f}, steps={episode_length}")

        # Save final evaluation results
        avg_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        avg_length = np.mean(eval_lengths)

        print(f"\nFinal evaluation results:")
        print(f"  Avg reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"  Avg episode length: {avg_length:.0f}")

        # Save summary
        with open(os.path.join(final_eval_dir, "summary.txt"), "w") as f:
            f.write(f"Final Evaluation on Test Data\n")
            f.write(f"{'='*40}\n")
            f.write(f"Number of episodes: {n_episodes}\n")
            f.write(f"Average reward: {avg_reward:.2f}\n")
            f.write(f"Std reward: {std_reward:.2f}\n")
            f.write(f"Average episode length: {avg_length:.0f}\n\n")
            f.write(f"Per-episode rewards:\n")
            for i, (r, l) in enumerate(zip(eval_rewards, eval_lengths)):
                f.write(f"  Episode {i}: reward={r:.2f}, length={l}\n")

        # Save rewards as CSV
        np.savetxt(
            os.path.join(final_eval_dir, "eval_rewards.csv"),
            eval_rewards,
            delimiter=";",
        )
        np.savetxt(
            os.path.join(final_eval_dir, "eval_lengths.csv"),
            eval_lengths,
            delimiter=";",
        )

        print(f"Final evaluation results saved to: {final_eval_dir}")
