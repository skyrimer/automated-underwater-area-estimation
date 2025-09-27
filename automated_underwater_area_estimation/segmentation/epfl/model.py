from typing import Literal, Tuple
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from transformers.image_processing_base import BatchFeature

from automated_underwater_area_estimation.segmentation.epfl.classmap import (
    EPFLClassMapping,
)
from automated_underwater_area_estimation.segmentation.model import (
    SegmentationModelBase,
)
from automated_underwater_area_estimation.segmentation.utils import get_best_device
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F

EPFLModelName = Literal[
    "EPFL-ECEO/segformer-b5-finetuned-coralscapes-1024-1024",
    "EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024",
]

EPFL_model_names = [
    "EPFL-ECEO/segformer-b5-finetuned-coralscapes-1024-1024",
    "EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024",
]


class EPFLModel(SegmentationModelBase):
    """EPFL Segformer model for coral reef segmentation."""

    def __init__(
        self, hf_model_name: EPFLModelName, device: torch.device | None = None
    ) -> None:
        # Validate model name
        if hf_model_name not in EPFL_model_names:
            raise ValueError(
                f"Model name must be one of {EPFL_model_names}, got: {hf_model_name}"
            )

        # Set required attributes
        self.model_name: str = hf_model_name
        self.preprocessor: SegformerImageProcessor = (
            SegformerImageProcessor.from_pretrained(hf_model_name)
        )
        self.model: SegformerForSemanticSegmentation = (
            SegformerForSemanticSegmentation.from_pretrained(hf_model_name)
        )
        self.class_mapping: EPFLClassMapping = EPFLClassMapping()
        self.ideal_size: Tuple[int, int] = (1024, 1024)

        self.model.eval()
        self.device: torch.device = device or get_best_device()
        super().__init__(self.device)

    def preprocess(self, image: Image.Image) -> BatchFeature:
        """Preprocess image for EPFL model input."""
        return self.preprocessor(image, return_tensors="pt").to(self.device)

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
        Based on the reference implementation that works directly with torch tensors.

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

        h_stride = int((h_img - h_crop + h_grids - 1) / (h_grids - 1)) if h_grids > 1 else h_crop
        w_stride = int((w_img - w_crop + w_grids - 1) / (w_grids - 1)) if w_grids > 1 else w_crop

        # Get the actual number of classes from the model's output
        # Do a quick test inference to determine the correct number of classes
        with torch.no_grad():
            test_crop = img[:, :, :min(h_crop, h_img), :min(w_crop, w_img)]
            test_crop_array = test_crop.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            test_crop_pil = Image.fromarray(test_crop_array)
            test_inputs = self.preprocess(test_crop_pil)
            test_outputs = self.model(**test_inputs)
            num_classes = test_outputs.logits.shape[1]  # Get actual number of classes from model output

        # Initialize prediction accumulator and count matrix
        preds = torch.zeros((batch_size, num_classes, h_img, w_img), dtype=torch.float32, device=self.device)
        count_mat = torch.zeros((batch_size, 1, h_img, w_img), dtype=torch.float32, device=self.device)

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
                    # Convert crop tensor back to PIL for preprocessing
                    # crop_img shape: (1, 3, h_crop, w_crop)
                    crop_array = crop_img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    crop_pil = Image.fromarray(crop_array)

                    # Preprocess the crop
                    inputs = self.preprocess(crop_pil)
                    outputs = self.model(**inputs)

                # Get logits and resize to crop size
                resized_logits = F.interpolate(
                    outputs.logits, size=crop_img.shape[-2:], mode="bilinear", align_corners=False
                )

                # Move logits to device before padding
                resized_logits = resized_logits.to(self.device)

                # Pad the logits to fit into the full image predictions
                padded_logits = F.pad(
                    resized_logits,
                    (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2))
                )

                preds += padded_logits
                count_mat[:, :, y1:y2, x1:x2] += 1

        # Ensure no division by zero
        assert (count_mat == 0).sum() == 0, "Some pixels were not covered by any window"

        # Average overlapping predictions
        preds = preds / count_mat

        # Get class predictions (argmax across classes)
        preds = preds.argmax(dim=1)

        # Resize back to original image size
        preds = F.interpolate(
            preds.unsqueeze(0).float(),
            size=image.size[::-1],  # PIL size is (width, height), need (height, width)
            mode='nearest'
        )

        final_segments = preds.squeeze().long().to(self.device)

        # Convert to binary classification using class mapping
        binary_mask = torch.zeros_like(final_segments, dtype=torch.bool, device=self.device)
        for class_id in self.class_mapping.CORAL_CLASS_IDS:
            binary_mask = torch.logical_or(binary_mask, final_segments == class_id)

        return image, binary_mask

    def segment_image(
            self, image: Image.Image, adjust_size: bool = True, use_sliding_window: bool = False
    ) -> Tuple[Image.Image, torch.Tensor]:
        """
        Segment an image using the EPFL model with binary classification.

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

        inputs = self.preprocess(image)
        with torch.no_grad():
            outputs = self.model(**inputs)

        segments = self.preprocessor.post_process_semantic_segmentation(
            outputs, target_sizes=[(image.size[1], image.size[0])]
        )[0]

        # Convert to binary classification using class mapping
        # Create boolean mask: True for coral classes, False for background
        binary_mask = torch.zeros_like(segments, dtype=torch.bool)

        for class_id in self.class_mapping.CORAL_CLASS_IDS:
            binary_mask = torch.logical_or(binary_mask, segments == class_id)

        return image, binary_mask