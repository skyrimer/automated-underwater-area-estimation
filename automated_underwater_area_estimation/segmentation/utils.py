import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def get_best_device(verbose: bool = False, force_device: str = "") -> torch.device:
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
            device_info.append(f"CUDA available: {torch.cuda.get_device_name()}")
            device_info.append(
                f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
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


def plot_segmentation_results(
    image, segmentation_mask, title_prefix="Segmentation Results"
):
    """
    Plot the original image, segmentation mask, and their overlay.

    Args:
        image (PIL.Image): Original image
        segmentation_mask (torch.Tensor): Binary segmentation mask of same dimensions as image
        title_prefix (str): Prefix for plot titles
    """
    # Convert inputs to numpy arrays for plotting
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image

    # Convert mask to numpy array and ensure it's on CPU
    if isinstance(segmentation_mask, torch.Tensor):
        mask_array = segmentation_mask.cpu().numpy().astype(bool)
    else:
        mask_array = segmentation_mask.astype(bool)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Original image
    axes[0].imshow(img_array)
    axes[0].set_title(f"{title_prefix} - Original Image")
    axes[0].axis("off")

    # Plot 2: Segmentation mask
    axes[1].imshow(mask_array, cmap="gray")
    axes[1].set_title(f"{title_prefix} - Segmentation Mask")
    axes[1].axis("off")

    # Plot 3: Overlay with 50% opacity
    axes[2].imshow(img_array)
    # Create colored mask (red for detected regions)
    colored_mask = np.zeros((*mask_array.shape, 4))  # RGBA
    colored_mask[mask_array] = [1, 0, 0, 0.5]  # Red with 50% opacity
    axes[2].imshow(colored_mask)
    axes[2].set_title(f"{title_prefix} - Overlay (50% opacity)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
