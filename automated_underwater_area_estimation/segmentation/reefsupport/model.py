from typing import Tuple, List, Literal, Any
from PIL import Image
import torch
import numpy as np
from automated_underwater_area_estimation.segmentation.model import (
    SegmentationModelBase,
)
from automated_underwater_area_estimation.segmentation.reefsupport.classmap import (
    ReefSupportClassMapping,
)
from automated_underwater_area_estimation.segmentation.utils import get_best_device
from ultralytics import YOLO
from pathlib import Path

# Type alias for valid ReefSupport model names
ReefSupportModelName = Literal["yolov8_sm_latest.pt", "yolov8_xlarge_latest.pt"]

REEFSUPPORT_model_names: List[str] = ["yolov8_sm_latest.pt", "yolov8_xlarge_latest.pt"]


class ReefSupportModel(SegmentationModelBase):
    """ReefSupport YOLO model for coral reef segmentation."""

    def __init__(
        self, model_name: ReefSupportModelName, device: torch.device | None = None
    ) -> None:
        # Validate model name
        if model_name not in REEFSUPPORT_model_names:
            raise ValueError(
                f"Model name must be one of {REEFSUPPORT_model_names}, got: {model_name}"
            )

        # Set required attributes
        self.model_name: str = model_name
        self.model_path: str = self.download_model(model_name)  # Use the returned path
        self.model: YOLO = YOLO(self.model_path, verbose=True)

        self.preprocessor: None = None  # YOLO handles preprocessing internally
        self.class_mapping: ReefSupportClassMapping = ReefSupportClassMapping()
        self.ideal_size: Tuple[int, int] = (1024, 1024)  # Standard YOLO input size
        self.device: torch.device = device or get_best_device()
        # Call parent validation
        super().__init__(self.device)

    def preprocess(self, image: Image.Image) -> Any:
        """YOLO handles preprocessing internally, so we just return the image as numpy array."""
        return np.array(image)

    @staticmethod
    def download_model(model_name: str) -> str:
        """Download a ReefSupport model if it doesn't exist locally in the same directory as the model file."""

        # Get the directory where the model.py file is located
        current_file_dir = Path(__file__).parent
        model_dir = current_file_dir / "models"

        # Create models directory if it doesn't exist
        model_dir.mkdir(exist_ok=True)

        # Full path to the model file
        model_path = model_dir / model_name

        if not model_path.exists():
            hf_model_repo = (
                "https://huggingface.co/reefsupport/coral-ai/resolve/main/models"
            )
            print(f"Downloading {model_name}...")
            torch.hub.download_url_to_file(
                f"{hf_model_repo}/{model_name}", str(model_path)
            )
            print(f"Downloaded {model_name} to {model_path}")

        return str(model_path)

    def resize_image(self, image: Image.Image, target_size: int = 1024) -> Image.Image:
        """
        Used to resize the image such that the smaller side equals target_size
        """
        w_img, h_img = image.size  # PIL format: (width, height)
        if w_img < h_img:
            new_w, new_h = target_size, int(h_img * (target_size / w_img))
        else:
            new_w, new_h = int(w_img * (target_size / h_img)), target_size
        resized_img = image.resize((new_w, new_h))
        return resized_img

    def segment_image_sliding_window(
        self, image: Image.Image, crop_size: Tuple[int, int] = None
    ) -> Tuple[Image.Image, torch.Tensor]:
        """
        Segment a high-resolution image using sliding window approach.
        Based on the tensor-based methodology for consistent processing.

        Args:
            image: Input PIL image
            crop_size: Size of each window (height, width). Defaults to ideal_size reversed.

        Returns:
            Tuple of (original_image, segmentation_map)
        """
        if crop_size is None:
            # Convert from PIL format (width, height) to tensor format (height, width)
            crop_size = (self.ideal_size[1], self.ideal_size[0])

        h_crop, w_crop = crop_size  # Tensor format: (height, width)

        # Resize image maintaining aspect ratio with smaller side = 1024
        resized_img = self.resize_image(image, target_size=1024)

        # Convert to tensor format: (1, C, H, W)
        img = torch.Tensor(np.array(resized_img).transpose(2, 0, 1)).unsqueeze(0)
        batch_size, _, h_img, w_img = img.size()

        # Move to device
        img = img.to(self.device)

        # Calculate grid dimensions and stride using 1.5x factor like the reference
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
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                # Extract crop
                crop_img = img[:, :, y1:y2, x1:x2]

                with torch.no_grad():
                    # Convert crop tensor back to PIL for YOLO processing
                    crop_array = (
                        crop_img.squeeze(0)
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    )
                    crop_pil = Image.fromarray(crop_array)

                    # Resize to ideal size for YOLO inference
                    crop_resized = crop_pil.resize(self.ideal_size)
                    crop_processed = self.preprocess(crop_resized)

                    # Run YOLO inference
                    results = self.model(crop_processed)

                # Process YOLO results to create binary mask for this crop
                crop_binary_mask = torch.zeros(
                    crop_img.shape[-2:], dtype=torch.float32, device=self.device
                )

                if (
                    len(results) > 0
                    and hasattr(results[0], "masks")
                    and results[0].masks is not None
                ):
                    masks = results[0].masks.data

                    if len(masks) > 0:
                        for i, mask in enumerate(masks):
                            # Get class ID for this detection
                            class_id = int(results[0].boxes.cls[i].item()) + 1

                            # Check if this is a coral class
                            if class_id in self.class_mapping.CORAL_CLASS_IDS:
                                # Resize mask to crop size and move to device
                                mask_resized = (
                                    torch.nn.functional.interpolate(
                                        mask.unsqueeze(0).unsqueeze(0).float(),
                                        size=crop_img.shape[-2:],
                                        mode="nearest",
                                    )
                                    .squeeze()
                                    .float()
                                    .to(self.device)
                                )

                                # Accumulate coral predictions (using max for overlapping detections)
                                crop_binary_mask = torch.maximum(
                                    crop_binary_mask, mask_resized
                                )

                # Add crop prediction to global prediction map using padding
                crop_binary_mask_batch = crop_binary_mask.unsqueeze(0).unsqueeze(
                    0
                )  # Add batch and channel dims

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
            size=image.size[::-1],  # PIL size is (width, height), need (height, width)
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
        Segment an image using the ReefSupport YOLO model with binary classification.

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

        # Convert PIL to numpy array for YOLO
        image_array = self.preprocess(image)

        # Run YOLO inference
        results = self.model(image_array)

        # Extract segmentation masks
        if (
            len(results) > 0
            and hasattr(results[0], "masks")
            and results[0].masks is not None
        ):
            # Get the first result's masks
            masks = results[
                0
            ].masks.data  # Shape: [N, H, W] where N is number of detections

            if len(masks) > 0:
                # Create binary segmentation map (True/False)
                image_width, image_height = image.size
                segmentation_map = torch.zeros(
                    (image_height, image_width), dtype=torch.bool, device=self.device
                )

                for i, mask in enumerate(masks):
                    # Get class ID for this detection
                    class_id = (
                        int(results[0].boxes.cls[i].item()) + 1
                    )  # +1 because our mapping starts at 1

                    # Use class mapping to check if this is a coral class
                    if class_id in self.class_mapping.CORAL_CLASS_IDS:
                        # Resize mask to match image size and apply
                        mask_resized = (
                            torch.nn.functional.interpolate(
                                mask.unsqueeze(0).unsqueeze(0).float(),
                                size=(image_height, image_width),
                                mode="nearest",
                            )
                            .squeeze()
                            .bool()
                        )
                        # Set True for coral pixels (binary OR operation)
                        segmentation_map = torch.logical_or(
                            segmentation_map, mask_resized
                        )

                return image, segmentation_map
            else:
                # No masks detected, return all False
                image_width, image_height = image.size
                segmentation_map = torch.zeros(
                    (image_height, image_width), dtype=torch.bool, device=self.device
                )
                return image, segmentation_map
        else:
            # No masks detected, return all False
            image_width, image_height = image.size
            segmentation_map = torch.zeros(
                (image_height, image_width), dtype=torch.bool, device=self.device
            )
            return image, segmentation_map
