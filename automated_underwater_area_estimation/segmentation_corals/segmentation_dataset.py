from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class CoralSegmentationDataset(Dataset):
    """
    PyTorch Dataset for coral segmentation data

    This dataset loads preprocessed coral images and their corresponding binary masks.
    Images are returned as PIL Images, masks as boolean tensors.
    """

    def __init__(
        self,
        data_root: str,
    ):
        """
        Initialize the coral segmentation dataset

        Args:
            data_root: Root directory containing 'images' and 'masks' subdirectories
            mask_transform: Optional transform to apply to masks
            return_filename: Whether to return filenames along with data
        """
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "images"
        self.masks_dir = self.data_root / "masks"

        # Validate directories exist
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        # Load samples
        self.samples = self._load_samples()

        if len(self.samples) == 0:
            raise ValueError(f"No valid image-mask pairs found in {data_root}")

    def _load_samples(self) -> List[Tuple[Path, Path]]:
        """Find and validate image-mask pairs"""
        samples = []

        # Get all mask files
        mask_files = list(self.masks_dir.glob("*.npy"))

        for mask_file in mask_files:
            base_name = mask_file.stem  # filename without extension

            # Find corresponding image file
            image_file = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                potential_image = self.images_dir / (base_name + ext)
                if potential_image.exists():
                    image_file = potential_image
                    break

            if image_file is not None:
                samples.append((image_file, mask_file))
            else:
                print(
                    f"Warning: No corresponding image found for mask {mask_file.name}"
                )

        return sorted(samples)  # Sort for consistent ordering

    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, torch.Tensor]:
        """
        Get a sample from the dataset

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (image, mask) or (image, mask, filename) if return_filename=True
            - image: PIL Image
            - mask: Boolean tensor (0/1 values)
            - filename: Original image filename (optional)
        """
        if idx >= len(self.samples):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.samples)}"
            )

        image_path, mask_path = self.samples[idx]

        # Load image as PIL Image
        image = Image.open(image_path).convert("RGB")

        # Load mask as numpy array and convert to boolean tensor
        mask_array = np.load(mask_path)
        mask_tensor = torch.from_numpy(mask_array).bool()

        return image, mask_tensor

    def visualize_sample(self, idx: int, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Visualize a sample (requires matplotlib)

        Args:
            idx: Index of sample to visualize
            figsize: Figure size for matplotlib
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualization")

        image, mask = self[idx]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Mask
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Segmentation Mask")
        axes[1].axis("off")

        # Overlay
        overlay = np.array(image)
        mask_np = mask.numpy()
        overlay[mask_np == 1] = [255, 0, 0]  # Red overlay for mask areas
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()
