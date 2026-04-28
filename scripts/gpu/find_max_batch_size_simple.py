#!/usr/bin/env python3
"""Find the maximum safe batch size for your GPU without crashing."""
import torch
import torch.nn as nn

print("=" * 70)
print("BATCH SIZE OPTIMIZER FOR H200")
print("=" * 70)

# Check GPU
if not torch.cuda.is_available():
    print("CUDA not available.")
    exit(1)

gpu_name = torch.cuda.get_device_name(0)
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

print(f"\nGPU: {gpu_name}")
print(f"Total Memory: {gpu_memory:.2f} GB")

# Create a model similar to your MAPPO networks
class TestCritic(nn.Module):
    """Simulates your critic network."""
    def __init__(self, hidden_size=512):
        super().__init__()
        input_size = 100  # Approximate your observation size
        self.fc1 = nn.Linear(input_size * 8, hidden_size)  # 8 agents
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        return self.value_head(x)

class TestActor(nn.Module):
    """Simulates your actor network."""
    def __init__(self, hidden_size=128):
        super().__init__()
        input_size = 50  # Approximate
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, 10)  # ~10 actions
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        return torch.softmax(self.action_head(x), dim=-1)

print("\n" + "=" * 70)
print("CREATING TEST MODELS")
print("=" * 70)

hidden_size = 512  # Your config value
n_agents = 8

# Create models on GPU
critic = TestCritic(hidden_size).cuda()
actors = [TestActor(128).cuda() for _ in range(n_agents)]

# Create optimizers
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.0003)
actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=0.0003) for actor in actors]

print(f"Created critic with hidden_size={hidden_size}")
print(f"Created {n_agents} actors")

# Test function
def test_batch_size(batch_size, verbose=False):
    """Test if a batch size works without OOM."""
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        if verbose:
            print(f"\n  Testing batch_size={batch_size}...", end=" ", flush=True)

        # Simulate critic training
        x_critic = torch.randn(batch_size, 100 * n_agents, device='cuda')
        targets = torch.randn(batch_size, 1, device='cuda')

        values = critic(x_critic)
        loss = ((values - targets) ** 2).mean()

        critic_optimizer.zero_grad()
        loss.backward()
        critic_optimizer.step()

        # Simulate actor training (for each agent)
        for i, (actor, optimizer) in enumerate(zip(actors, actor_optimizers)):
            x_actor = torch.randn(batch_size, 50, device='cuda')
            probs = actor(x_actor)
            loss = -probs.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9

        if verbose:
            print(f"Peak: {peak:.2f} GB")

        return True, peak

    except RuntimeError as e:
        if "out of memory" in str(e):
            if verbose:
                print(f"OOM")
            torch.cuda.empty_cache()
            return False, None
        else:
            raise e

# Find maximum batch size
print("\n" + "=" * 70)
print("SEARCHING FOR MAXIMUM BATCH SIZE")
print("=" * 70)

test_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 49152, 65536]
last_working = 1024
last_peak = 0

print("\nTesting batch sizes:")
for size in test_sizes:
    success, peak = test_batch_size(size, verbose=True)
    if success:
        last_working = size
        last_peak = peak
    else:
        break

# Binary search for exact max
if last_working < test_sizes[-1]:
    print("\nFine-tuning with binary search...")
    min_batch = last_working
    max_batch = last_working * 2

    for _ in range(5):
        mid = (min_batch + max_batch) // 2
        mid = (mid // 512) * 512  # Round to 512

        if mid == min_batch:
            break

        success, peak = test_batch_size(mid, verbose=True)
        if success:
            min_batch = mid
            last_working = mid
            last_peak = peak
        else:
            max_batch = mid

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

max_safe = last_working
recommended = int(max_safe * 0.8)
recommended = (recommended // 512) * 512

print(f"\nMaximum batch size: {max_safe}")
print(f"   Peak memory: {last_peak:.2f} GB / {gpu_memory:.2f} GB")
print(f"\nRECOMMENDED: {recommended}")
print(f"   (80% of max for safety)")

# Calculate impact
current_batch = 8192  # Your current setting
n_samples = 62510

batches_current = (n_samples + current_batch - 1) // current_batch
batches_recommended = (n_samples + recommended - 1) // recommended

print(f"\nIMPACT:")
print(f"   Current ({current_batch}): {batches_current} batches/epoch")
print(f"   Recommended ({recommended}): {batches_recommended} batches/epoch")

if recommended > current_batch:
    speedup = batches_current / batches_recommended
    print(f"   Expected speedup: ~{speedup:.2f}x")
else:
    print(f"   Current batch size is already optimal.")

print("\n" + "=" * 70)
print("UPDATE YOUR CONFIG")
print("=" * 70)
print(f"\nEdit: src/configs/phase_b_standard_rl/mappo_baseline.yaml")
print(f"  batch_size: {recommended}")
print(f"  buffer_size: {recommended}")

print("\nDone.")
