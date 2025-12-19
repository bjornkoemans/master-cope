"""
Hyperparameter configuration for MAPPO training.
Modify these values to experiment with different settings.
"""

# Learning rates
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 3e-4

# Discount factor
GAMMA = 0.99

# Network architecture
ACTOR_HIDDEN_SIZE = 128
CRITIC_HIDDEN_SIZE = 256
NUM_HIDDEN_LAYERS = 3
DROPOUT_RATE = 0.2

# PPO parameters
PPO_CLIP_PARAM = 0.2
GAE_LAMBDA = 0.95

# Batch size configuration
# Note: When AUTO_BATCH_SIZE=True, this serves as a fallback/max value
# The actual batch size will be automatically determined based on:
# - Number of samples in episode
# - Available GPU memory
# - Model size
BATCH_SIZE = 32768  # Fallback value
AUTO_BATCH_SIZE = True  # Enable automatic batch size optimization

# Training epochs (how many times to iterate over collected data)
NUM_EPOCHS = 10

# Weight initialization
# Options: "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "orthogonal"
WEIGHT_INIT = "xavier_uniform"

# Training settings
DEFAULT_TRAINING_EPISODES = 50
DEFAULT_POLICY_UPDATE_EPOCHS = 5

# GPU Optimization settings
USE_MIXED_PRECISION = True  # Enable FP16/BF16 training on compatible GPUs
# Mixed precision provides 2-3x speedup on NVIDIA H200, A100, and newer GPUs

# QMIX specific parameters
QMIX_LEARNING_RATE = 5e-4
QMIX_EPSILON = 0.1
QMIX_BUFFER_SIZE = 10000
QMIX_TARGET_UPDATE_INTERVAL = 100
QMIX_BATCH_SIZE = 32
