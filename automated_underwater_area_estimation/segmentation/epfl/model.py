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

        self.device: torch.device = device or get_best_device()
        super().__init__(self.device)

    def segment_image(
        self, image: Image.Image, adjust_size: bool = True
    ) -> Tuple[Image.Image, torch.Tensor]:
        """Segment an image using the EPFL model with binary classification."""
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

    def preprocess(self, image: Image.Image) -> BatchFeature:
        """Preprocess image for EPFL model input."""
        return self.preprocessor(image, return_tensors="pt").to(self.device)
