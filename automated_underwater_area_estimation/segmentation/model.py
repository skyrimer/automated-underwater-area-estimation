from abc import ABC, abstractmethod
from typing import Tuple, Any
from PIL import Image
import torch
from automated_underwater_area_estimation.segmentation.class_mapping import (
    ClassMappingBase,
)


class SegmentationModelBase(ABC):
    """
    Simple abstract base class for segmentation models.
    Enforces the essential attributes and methods that all models must have.
    """

    def __init__(self, device: torch.device):
        # Validate that required attributes are present after initialization
        self._validate_required_attributes()

        # Move model to device if it exists and has a 'to' method
        if hasattr(self, "model") and hasattr(self.model, "to"):
            self.model.to(device)

    def _validate_required_attributes(self) -> None:
        """Check that all required attributes are present."""
        required_attrs = [
            "model_name",
            "preprocessor",
            "model",
            "class_mapping",
            "ideal_size",
        ]

        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(f"Model must have '{attr}' attribute")

        # Validate ideal_size format
        if (
            not isinstance(self.ideal_size, tuple)
            or len(self.ideal_size) != 2
            or not all(isinstance(x, int) and x > 0 for x in self.ideal_size)
        ):
            raise ValueError("ideal_size must be a tuple of two positive integers")

        # Validate class_mapping type
        if not isinstance(self.class_mapping, ClassMappingBase):
            raise TypeError("class_mapping must inherit from ClassMappingBase")

    @abstractmethod
    def segment_image(
        self, image: Image.Image, adjust_size: bool = True
    ) -> Tuple[Image.Image, torch.Tensor]:
        """
        Segment an input image.

        Args:
            image: PIL Image to segment
            adjust_size: Whether to resize image to ideal size

        Returns:
            Tuple of (processed_image, segmentation_mask)
        """
        pass

    @abstractmethod
    def preprocess(self, image: Image.Image) -> Any:
        """
        Preprocess image for model input.

        Args:
            image: PIL Image to preprocess

        Returns:
            Preprocessed input ready for model
        """
        pass

    def get_class_name(self, class_id: int) -> str:
        """Get class name from class ID."""
        return self.class_mapping.get_name(class_id)

    def get_all_class_names(self) -> list[str]:
        """Get all class names."""
        return self.class_mapping.get_all_names()

    def get_num_classes(self) -> int:
        """Get number of classes."""
        return self.class_mapping.get_num_classes()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name})"

    def __repr__(self) -> str:
        return self.__str__()
