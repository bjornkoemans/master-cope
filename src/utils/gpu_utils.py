"""
GPU Optimization Utilities for NVIDIA H200 and other GPUs.

This module provides:
- Automatic batch size optimization based on available GPU memory
- Mixed precision training setup
- GPU utilization monitoring
"""

import torch
import numpy as np
from typing import Optional


def get_optimal_batch_size(
    num_samples: int,
    device: torch.device,
    model_size_mb: Optional[float] = None,
    safety_factor: float = 0.7,
) -> int:
    """
    Automatically determine optimal batch size based on:
    - Number of samples in episode
    - Available GPU memory
    - Model size

    Args:
        num_samples: Number of timesteps in the episode buffer
        device: torch.device to use
        model_size_mb: Estimated model size in MB (optional)
        safety_factor: Use this fraction of available memory (default 0.7)

    Returns:
        Optimal batch size
    """
    if device.type == "cpu":
        # For CPU, use smaller batches to avoid memory issues
        return min(num_samples, 512)

    if device.type == "cuda":
        try:
            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(device).total_memory
            reserved_memory = torch.cuda.memory_reserved(device)
            allocated_memory = torch.cuda.memory_allocated(device)
            free_memory = total_memory - max(reserved_memory, allocated_memory)

            # Convert to MB
            free_memory_mb = free_memory / (1024 ** 2)

            # Estimate memory per sample (rough heuristic)
            if model_size_mb is None:
                # Assume ~4 bytes per parameter + activation memory
                # For typical RL observation ~100 floats = 400 bytes
                # Plus gradients and optimizer states
                memory_per_sample = 0.002  # MB (2 KB per sample, conservative)
            else:
                memory_per_sample = model_size_mb / 1000

            # Calculate max batch size that fits in memory
            max_batch_size = int((free_memory_mb * safety_factor) / memory_per_sample)

            # Ensure batch size is reasonable
            # Min: 32, Max: num_samples or 65536
            optimal_batch = max(32, min(max_batch_size, num_samples, 65536))

            # Round down to nearest power of 2 for efficiency
            optimal_batch = 2 ** int(np.log2(optimal_batch))

            return optimal_batch

        except Exception as e:
            print(f"Warning: Could not determine optimal batch size: {e}")
            return min(num_samples, 2048)

    elif device.type == "mps":
        # For Apple Silicon, use moderate batch sizes
        return min(num_samples, 2048)

    return min(num_samples, 1024)


def get_gpu_memory_info(device: torch.device) -> dict:
    """Get detailed GPU memory information."""
    if device.type != "cuda":
        return {"type": str(device.type), "available": False}

    try:
        props = torch.cuda.get_device_properties(device)
        total = torch.cuda.get_device_properties(device).total_memory
        reserved = torch.cuda.memory_reserved(device)
        allocated = torch.cuda.memory_allocated(device)
        free = total - max(reserved, allocated)

        return {
            "name": props.name,
            "total_gb": total / (1024 ** 3),
            "reserved_gb": reserved / (1024 ** 3),
            "allocated_gb": allocated / (1024 ** 3),
            "free_gb": free / (1024 ** 3),
            "compute_capability": f"{props.major}.{props.minor}",
        }
    except Exception as e:
        return {"error": str(e)}


def setup_mixed_precision(device: torch.device, enabled: bool = True):
    """
    Setup mixed precision training (FP16/BF16) for faster training.

    Args:
        device: torch.device to use
        enabled: Whether to enable mixed precision

    Returns:
        (scaler, dtype) tuple for mixed precision training
    """
    if not enabled or device.type == "cpu":
        return None, torch.float32

    if device.type == "cuda":
        # Check compute capability
        compute_cap = torch.cuda.get_device_properties(device).major

        # H200 and A100+ support BF16 (better than FP16 for training)
        if compute_cap >= 8:  # Ampere, Hopper, etc.
            # Use BF16 for better numerical stability
            dtype = torch.bfloat16
            scaler = torch.cuda.amp.GradScaler(enabled=False)  # BF16 doesn't need scaling
            print(f"🚀 Mixed Precision: BF16 enabled (Compute {compute_cap}.x)")
        else:
            # Use FP16 with gradient scaling
            dtype = torch.float16
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            print(f"🚀 Mixed Precision: FP16 enabled with gradient scaling")

        return scaler, dtype

    elif device.type == "mps":
        # Apple Silicon supports FP16
        return None, torch.float16

    return None, torch.float32


def print_gpu_utilization(device: torch.device):
    """Print current GPU utilization."""
    if device.type == "cuda":
        info = get_gpu_memory_info(device)
        if "error" not in info:
            print(f"\n📊 GPU Utilization ({info['name']}):")
            print(f"   Total Memory: {info['total_gb']:.2f} GB")
            print(f"   Allocated:    {info['allocated_gb']:.2f} GB ({info['allocated_gb']/info['total_gb']*100:.1f}%)")
            print(f"   Free:         {info['free_gb']:.2f} GB ({info['free_gb']/info['total_gb']*100:.1f}%)")
        else:
            print(f"⚠️  Could not get GPU info: {info['error']}")


def optimize_for_h200(device: torch.device) -> dict:
    """
    Apply H200-specific optimizations.

    Returns:
        Dictionary with optimization settings
    """
    if device.type != "cuda":
        return {"optimized": False}

    try:
        props = torch.cuda.get_device_properties(device)

        # Enable TF32 for H200/A100+ (faster matmul)
        if props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Enable cudnn benchmarking for optimal kernels
        torch.backends.cudnn.benchmark = True

        # Optimize memory allocator
        torch.cuda.empty_cache()

        return {
            "optimized": True,
            "tf32_enabled": props.major >= 8,
            "cudnn_benchmark": True,
            "device": props.name,
        }
    except Exception as e:
        return {"optimized": False, "error": str(e)}


def estimate_model_size(model: torch.nn.Module) -> float:
    """
    Estimate model size in MB.

    Args:
        model: PyTorch model

    Returns:
        Estimated size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb
