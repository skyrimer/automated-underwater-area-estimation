"""
CoralSCOP Model - A foundation model for coral segmentation

This module provides a clean interface to the CoralSCOP model for coral reef segmentation.
The model uses a ViT-B backbone with default parameters and provides methods for
coral segmentation from PIL images.
"""

import os
import ssl
import urllib.request

import cv2
import numpy as np
import torch
from PIL import Image

from local_segment_anything import SamAutomaticMaskGenerator, build_sam_vit_b


class CoralSCOP:
    """
    CoralSCOP model for coral reef segmentation.

    This class provides a simple interface to the CoralSCOP foundation model
    for automatic coral segmentation. It initializes the model with ViT-B backbone
    and default parameters, automatically handles device placement, and provides
    methods for segmenting coral reefs in images.
    """

    # Default checkpoint URL
    DEFAULT_CHECKPOINT_URL = "https://www.dropbox.com/scl/fi/pw5jiq9oc8e8kvkx1fdk0/vit_b_coralscop.pth?rlkey=qczdohnzxwgwoadpzeht0lim2&e=2&st=actcedwy&dl=1"

    @staticmethod
    def _download_checkpoint(url: str, output_path: str) -> None:
        """
        Download the model checkpoint from the given URL.

        Args:
            url (str): URL to download the checkpoint from
            output_path (str): Path where to save the checkpoint
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"Downloading checkpoint from {url}")
        print("This may take a few minutes...")

        # Create SSL context that doesn't verify certificates (for some download services)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        def download_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size / total_size) * 100)
                print(
                    f"\rDownload progress: {percent:.1f}%", end="", flush=True)

        try:
            # Download with progress
            urllib.request.urlretrieve(
                url, output_path, reporthook=download_progress)
            print(f"\nCheckpoint downloaded successfully to {output_path}")
        except Exception as e:
            print(f"\nError downloading checkpoint: {e}")
            print("Please download the checkpoint manually from:")
            print("https://www.dropbox.com/scl/fi/pw5jiq9oc8e8kvkx1fdk0/vit_b_coralscop.pth?rlkey=qczdohnzxwgwoadpzeht0lim2&e=2&st=actcedwy&dl=0")
            raise

    def __init__(
        self,
        checkpoint_path: str = "./checkpoints/vit_b_coralscop.pth",
        device: str | None = None,
    ):
        """
        Initialize the CoralSCOP model.

        Args:
            checkpoint_path (str): Path to the model checkpoint file
            device (str, optional): Device to run the model on ('cuda', 'cpu', or None for auto)
            points_per_side (int): Number of points per side for automatic mask generation
            pred_iou_thresh (float): IoU threshold for predicted masks
            stability_score_thresh (float): Stability score threshold for masks
            crop_n_layers (int): Number of crop layers for processing
            crop_n_points_downscale_factor (int): Downscale factor for crop points
            min_mask_region_area (int): Minimum area for mask regions
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Check if checkpoint exists, download if not
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}")
            if checkpoint_path == "./checkpoints/vit_b_coralscop.pth":
                # Download the default checkpoint
                self._download_checkpoint(
                    self.DEFAULT_CHECKPOINT_URL, checkpoint_path)
            else:
                raise FileNotFoundError(
                    f"Checkpoint file not found: {checkpoint_path}")

        self.checkpoint_path = checkpoint_path
        print(f"Initializing CoralSCOP model on {self.device}...")

        # Load the SAM model with vit_b backbone
        self.sam = build_sam_vit_b(checkpoint=self.checkpoint_path)
        self.sam.to(device=self.device)
        self.sam.eval()

        # Initialize the automatic mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
        )

        print("CoralSCOP model initialized successfully!")

    def segment_image(self, image: Image.Image) -> torch.Tensor:
        """
        Segment coral reefs in the input image.

        Args:
            image (PIL.Image): Input image as PIL Image

        Returns:
            torch.Tensor: Binary mask tensor of the same spatial shape as input image,
                         where coral pixels are marked as 1 and non-coral as 0.
                         Shape: (H, W) with dtype torch.float32
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            # Convert RGBA to RGB if needed
            if image_np.shape[-1] == 4:
                image_np = image_np[:, :, :3]
        else:
            image_np = image.copy()

        if len(image_np.shape) != 3 or image_np.shape[-1] != 3:
            raise ValueError(
                f"Unsupported image format. Expected (H, W, 3), got {image_np.shape}")

        # If image is BGR (common with OpenCV), convert to RGB
        if hasattr(image, 'mode') and image.mode == 'BGR':
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        # Get image dimensions
        height, width = image_np.shape[:2]

        # Generate masks using the automatic mask generator
        with torch.no_grad():
            try:
                masks = self.mask_generator.generate(image_np)
            except Exception as e:
                print(f"Error during mask generation: {e}")
                # Return empty mask on error
                return torch.zeros((height, width), dtype=torch.float32)

        # Combine all masks into a single binary mask
        combined_mask = np.zeros((height, width), dtype=np.float32)

        for mask_data in masks:
            # Extract the segmentation mask
            segmentation = mask_data['segmentation']

            # Add to combined mask (coral regions = 1)
            combined_mask = np.logical_or(
                combined_mask, segmentation).astype(np.float32)

        return torch.from_numpy(combined_mask)

    def get_detailed_masks(self, image: Image.Image) -> list:
        """
        Get detailed mask information including individual coral segments.

        Args:
            image (PIL.Image): Input image

        Returns:
            list: List of dictionaries containing detailed mask information
                  Each dict contains: segmentation, bbox, area, predicted_iou,
                  stability_score, point_coords, crop_box
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            if image_np.shape[-1] == 4:
                image_np = image_np[:, :, :3]
        else:
            image_np = image.copy()

        if hasattr(image, 'mode') and image.mode == 'BGR' and (len(image_np.shape) == 3 and image_np.shape[-1] == 3):
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # Generate detailed masks
        with torch.no_grad():
            masks = self.mask_generator.generate(image_np)

        return masks

    def __repr__(self):
        """String representation of the model."""
        return (f"CoralSCOP(model_type='vit_b', "
                f"device='{self.device}', "
                f"checkpoint='{os.path.basename(self.checkpoint_path)}')")

    def to(self, device: str):
        """Move model to specified device."""
        self.device = device
        self.sam.to(device)
        return self


# Example usage
if __name__ == "__main__":
    # Initialize the model
    model = CoralSCOP()

    # Load a test image
    from PIL import Image
    image_path = "./demo_imgs/AUSTRALES_AUS_RAI1###20M###20190504_UTP_QUADRAT_AUSRAI1_20M_10.jpg"

    if os.path.exists(image_path):
        image = Image.open(image_path)

        # Segment the image
        coral_mask = model.segment_image(image)

        print(f"Image shape: {np.array(image).shape}")
        print(f"Mask shape: {coral_mask.shape}")
        print(
            f"Coral coverage: {coral_mask.sum().item() / coral_mask.numel() * 100:.2f}%")

        # Get detailed masks
        detailed_masks = model.get_detailed_masks(image)
        print(f"Number of individual coral segments: {len(detailed_masks)}")
    else:
        print(f"Test image not found at {image_path}")
        print("Please ensure you have demo images or modify the path.")
