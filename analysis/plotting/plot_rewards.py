import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_rewards(experiment_dir):
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Get all episode directories
    episodes_dir = Path(experiment_dir) / "episodes"
    episode_dirs = sorted([d for d in episodes_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])
    
    # Plot training rewards
    for episode_dir in episode_dirs:
        # Read rewards for this episode
        rewards_file = episode_dir / "rewards.csv"
        if rewards_file.exists():
            rewards = np.loadtxt(rewards_file, delimiter=";")
            # Calculate cumulative rewards
            cumulative_rewards = np.cumsum(rewards)
            # Plot with episode number in legend
            episode_num = int(episode_dir.name.split("_")[1])
            ax1.plot(cumulative_rewards, label=f'Episode {episode_num}')
    
    # Customize training rewards plot
    ax1.set_title('Cumulative Training Rewards per Episode')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Cumulative Reward')
    ax1.grid(True)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot evaluation rewards
    for episode_dir in episode_dirs:
        eval_dir = episode_dir / "evaluation"
        if eval_dir.exists():
            eval_rewards_file = eval_dir / "eval_reward.csv"
            if eval_rewards_file.exists():
                eval_reward = np.loadtxt(eval_rewards_file, delimiter=";")
                episode_num = int(episode_dir.name.split("_")[1])
                ax2.bar(episode_num, eval_reward, label=f'Episode {episode_num}')
    
    # Customize evaluation rewards plot
    ax2.set_title('Evaluation Rewards per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.grid(True)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(Path(experiment_dir) / 'reward_plots.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    # Use the experiment directory
    experiment_dir = "experiments/mappo_20250611_202135"
    plot_rewards(experiment_dir) 