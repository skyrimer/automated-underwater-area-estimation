import itertools
import os
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from tqdm import tqdm

from automated_underwater_area_estimation.segmentation_corals.coralscop.classmap import (
    CoralScopeClassMapping,
)
from automated_underwater_area_estimation.segmentation_corals.model import (
    SegmentationModelBase,
)
from automated_underwater_area_estimation.utils import (
    get_best_device,
)

from .local_segment_anything import SamAutomaticMaskGenerator, sam_model_registry


class CoralSCOP(SegmentationModelBase):
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

        try:
            # Send GET request with stream=True for large files
            response = requests.get(url, stream=True, verify=False)
            response.raise_for_status()

            # Get file size for progress bar
            total_size = int(response.headers.get("content-length", 0))

            # Download with tqdm progress bar
            with open(output_path, "wb") as file:
                with tqdm(
                    desc="Downloading checkpoint",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            pbar.update(len(chunk))

            print(f"Checkpoint downloaded successfully to {output_path}")

        except Exception as e:
            print(f"Error downloading checkpoint: {e}")
            print(f"Please download the checkpoint manually from: {url}")
            raise

    def __init__(
        self,
        checkpoint_path: str = "vit_b_coralscop.pth",
        device: torch.device | None = None,
    ):
        """
        Initialize the CoralSCOP model.

        Args:
            checkpoint_path (str): Path to the model checkpoint file
            device (torch.device, optional): Device to run the model on
        """
        # Set required attributes for base class
        self.model_name: str = "CoralSCOP_vit_b"
        self.preprocessor: None = None  # SAM handles preprocessing internally
        self.class_mapping: CoralScopeClassMapping = CoralScopeClassMapping()
        self.ideal_size: Tuple[int, int] = (1024, 1024)  # Standard input size

        # Auto-detect device if not specified
        self.device: torch.device = device or get_best_device()

        # Check if checkpoint exists, download if not
        # Navigate to the project root's checkpoints folder
        model_dir = Path(__file__).parent / "checkpoints"
        model_dir.mkdir(exist_ok=True)
        full_checkpoint_path = model_dir / checkpoint_path
        if not os.path.exists(full_checkpoint_path):
            print(f"Checkpoint not found at {full_checkpoint_path}")
            self._download_checkpoint(
                self.DEFAULT_CHECKPOINT_URL, str(full_checkpoint_path)
            )

        checkpoint_path = str(full_checkpoint_path)

        self.checkpoint_path = checkpoint_path
        print(f"Initializing CoralSCOP model on {self.device}...")

        # Load the SAM model with vit_b backbone
        print(self.checkpoint_path)
        self.model = sam_model_registry["vit_b"](checkpoint=self.checkpoint_path)
        self.model.to(device=self.device)
        self.model.eval()

        # Initialize the automatic mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.model,
        )

        print("CoralSCOP model initialized successfully!")

        # Call parent validation
        super().__init__(self.device)

    def preprocess(self, image: Image.Image) -> Any:
        """SAM handles preprocessing internally, so we convert PIL to numpy array."""
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            # Convert RGBA to RGB if needed
            if image_np.shape[-1] == 4:
                image_np = image_np[:, :, :3]
        else:
            image_np = image.copy()
        return image_np

    def resize_image(self, image: Image.Image, target_size: int = 1024) -> Image.Image:
        """
        Resize image maintaining aspect ratio with smaller dimension equal to target_size.
        
        Args:
            image: Input PIL Image to resize
            target_size: Target size for the smaller dimension (default: 1024)
            
        Returns:
            Resized PIL Image with smaller dimension equal to target_size
        """
        w_img, h_img = image.size  # PIL format: (width, height)
        if w_img < h_img:
            new_w, new_h = target_size, int(h_img * (target_size / w_img))
        else:
            new_w, new_h = int(w_img * (target_size / h_img)), target_size
        return image.resize((new_w, new_h))

    def segment_image_sliding_window(
        self, image: Image.Image
    ) -> Tuple[Image.Image, torch.Tensor]:
        """
        Segment a high-resolution image using sliding window approach.

        Args:
            image: Input PIL image
            crop_size: Size of each window (height, width). Defaults to ideal_size reversed.

        Returns:
            Tuple of (original_image, segmentation_map)
        """

        h_crop, w_crop = self.ideal_size[::-1]  # Tensor format: (height, width)

        # Resize image maintaining aspect ratio with smaller side = 1024
        resized_img = self.resize_image(image, target_size=1024)

        # Convert to tensor format: (1, C, H, W)
        img = torch.Tensor(np.array(resized_img).transpose(2, 0, 1)).unsqueeze(0)
        batch_size, _, h_img, w_img = img.size()

        # Move to device
        img = img.to(self.device)

        # Calculate grid dimensions and stride using 1.5x factor
        h_grids = int(np.round(1.5 * h_img / h_crop)) if h_img > h_crop else 1
        w_grids = int(np.round(1.5 * w_img / w_crop)) if w_img > w_crop else 1

        h_stride = (
            int((h_img - h_crop + h_grids - 1) / (h_grids - 1))
            if h_grids > 1
            else h_crop
        )
        w_stride = (
            int((w_img - w_crop + w_grids - 1) / (w_grids - 1))
            if w_grids > 1
            else w_crop
        )

        # Initialize prediction accumulator and count matrix for binary segmentation
        preds = torch.zeros(
            (batch_size, 1, h_img, w_img), dtype=torch.float32, device=self.device
        )
        count_mat = torch.zeros(
            (batch_size, 1, h_img, w_img), dtype=torch.float32, device=self.device
        )

        # Process each window
        for h_idx, w_idx in itertools.product(range(h_grids), range(w_grids)):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)

            # Extract crop
            crop_img = img[:, :, y1:y2, x1:x2]

            with torch.no_grad():
                # Convert crop tensor back to PIL for SAM processing
                crop_array = (
                    crop_img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                )
                crop_pil = Image.fromarray(crop_array)

                # Preprocess the crop
                crop_processed = self.preprocess(crop_pil)

                # Generate masks using SAM
                try:
                    masks = self.mask_generator.generate(crop_processed)
                except Exception as e:
                    print(f"Error during mask generation for crop: {e}")
                    masks = []

            # Process SAM results to create binary mask for this crop
            crop_binary_mask = torch.zeros(
                crop_img.shape[-2:], dtype=torch.float32, device=self.device
            )

            # Combine all masks into a single binary mask for this crop
            for mask_data in masks:
                segmentation = mask_data["segmentation"]
                # Convert to tensor and add to crop mask
                mask_tensor = torch.from_numpy(segmentation.astype(np.float32)).to(
                    self.device
                )
                crop_binary_mask = torch.maximum(crop_binary_mask, mask_tensor)

            # Add crop prediction to global prediction map using padding
            crop_binary_mask_batch = crop_binary_mask.unsqueeze(0).unsqueeze(0)

            # Pad the crop prediction to fit into the full image predictions
            padded_prediction = torch.nn.functional.pad(
                crop_binary_mask_batch,
                (
                    int(x1),
                    int(preds.shape[3] - x2),
                    int(y1),
                    int(preds.shape[2] - y2),
                ),
            )

            preds += padded_prediction
            count_mat[:, :, y1:y2, x1:x2] += 1

        # Ensure no division by zero
        assert (count_mat == 0).sum() == 0, "Some pixels were not covered by any window"

        # Average overlapping predictions
        preds = preds / count_mat

        # Convert to binary mask (threshold at 0.5)
        preds = (preds > 0.5).float()

        # Resize back to original image size
        preds = torch.nn.functional.interpolate(
            preds,
            # PIL size is (width, height), need (height, width)
            size=image.size[::-1],
            mode="nearest",
        )

        final_segmentation_map = preds.squeeze().bool().to(self.device)

        return image, final_segmentation_map

    def segment_image(
        self,
        image: Image.Image,
        adjust_size: bool = True,
        use_sliding_window: bool = False,
    ) -> Tuple[Image.Image, torch.Tensor]:
        """
        Segment coral reefs in the input image using the CoralSCOP model.

        Args:
            image: Input PIL image
            adjust_size: Whether to resize image (ignored if use_sliding_window=True)
            use_sliding_window: Whether to use sliding window approach for high-res images

        Returns:
            Tuple of (processed_image, segmentation_map)
        """
        if use_sliding_window:
            return self.segment_image_sliding_window(image)

        # Original implementation for backward compatibility
        if adjust_size:
            image = image.resize(self.ideal_size)

        # Convert PIL Image to numpy array
        image_np = self.preprocess(image)

        if len(image_np.shape) != 3 or image_np.shape[-1] != 3:
            raise ValueError(
                f"Unsupported image format. Expected (H, W, 3), got {image_np.shape}"
            )

        # If image is BGR (common with OpenCV), convert to RGB
        if hasattr(image, "mode") and image.mode == "BGR":
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
                empty_mask = torch.zeros(
                    (height, width), dtype=torch.bool, device=self.device
                )
                return image, empty_mask

        # Combine all masks into a single binary mask
        combined_mask = np.zeros((height, width), dtype=np.float32)

        for mask_data in masks:
            # Extract the segmentation mask
            segmentation = mask_data["segmentation"]

            # Add to combined mask (coral regions = 1)
            combined_mask = np.logical_or(combined_mask, segmentation).astype(
                np.float32
            )

        # Convert to boolean tensor and move to device
        binary_mask = torch.from_numpy(combined_mask.astype(bool)).to(self.device)

        return image, binary_mask

    def get_detailed_masks(self, image: Image.Image) -> list:
        """
        Get detailed mask information including individual coral segments.
        
        Generates segmentation masks using SAM and returns detailed information
        about each detected segment including bounding boxes, areas, and quality scores.

        Args:
            image: Input PIL Image to segment

        Returns:
            List of dictionaries, each containing:
                - segmentation: Binary mask for the segment
                - bbox: Bounding box coordinates [x, y, w, h]
                - area: Pixel area of the segment
                - predicted_iou: SAM's predicted IoU score
                - stability_score: Mask stability score
                - point_coords: Coordinates of sampled points
                - crop_box: Cropping box used for this segment
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            if image_np.shape[-1] == 4:
                image_np = image_np[:, :, :3]
        else:
            image_np = image.copy()

        if (
            hasattr(image, "mode")
            and image.mode == "BGR"
            and (len(image_np.shape) == 3 and image_np.shape[-1] == 3)
        ):
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # Generate detailed masks
        with torch.no_grad():
            masks = self.mask_generator.generate(image_np)

        return masks

    def __repr__(self):
        """String representation of the model."""
        return (
            f"CoralSCOP(model_type='vit_b', "
            f"device='{self.device}', "
            f"checkpoint='{os.path.basename(self.checkpoint_path)}')"
        )

    def to(self, device: torch.device | str):
        """Move model to specified device."""
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.model.to(device)
        return self


# Example usage
if __name__ == "__main__":
    # Initialize the model
    model = CoralSCOP()

    # Load a test image
    from PIL import Image

    image_path = (
        "./demo_imgs/AUSTRALES_AUS_RAI1###20M###20190504_UTP_QUADRAT_AUSRAI1_20M_10.jpg"
    )

    if os.path.exists(image_path):
        image = Image.open(image_path)

        # Segment the image
        processed_image, coral_mask = model.segment_image(image)

        print(f"Image shape: {np.array(image).shape}")
        print(f"Mask shape: {coral_mask.shape}")
        print(
            f"Coral coverage: {coral_mask.sum().item() / coral_mask.numel() * 100:.2f}%"
        )

        # Get detailed masks
        detailed_masks = model.get_detailed_masks(image)
        print(f"Number of individual coral segments: {len(detailed_masks)}")
    else:
        print(f"Test image not found at {image_path}")
        print("Please ensure you have demo images or modify the path.")
