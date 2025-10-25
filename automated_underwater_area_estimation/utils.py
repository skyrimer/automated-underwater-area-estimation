import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def get_best_device(verbose: bool = True, force_device: str = "") -> torch.device:
    """
    Get the best available device for PyTorch operations.
    Priority order: CUDA > MPS > XPU > CPU

    Args:
        verbose: If True, print information about the selected device
        force_device: If provided, force the use of the specified device

    Returns:
        torch.device: The best available device
    """
    if force_device:
        return torch.device(force_device)
    device_info = []

    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            device_info.extend(
                (
                    f"CUDA available: {torch.cuda.get_device_name()}",
                    f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1000000000.0:.1f} GB",
                )
            )
        return device

    # Check for MPS (Apple Silicon M1/M2/M3 GPUs)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            device_info.append("MPS (Apple Silicon GPU) available")
        return device

    # Check for XPU (Intel GPUs)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        if verbose:
            device_info.append("XPU (Intel GPU) available")
        return device

    # Fallback to CPU
    device = torch.device("cpu")
    if verbose:
        device_info.append("Using CPU")
        device_info.append(f"CPU cores: {torch.get_num_threads()}")

    if verbose and device_info:
        print(f"Selected device: {device}")
        for info in device_info:
            print(f"  {info}")

    return device
