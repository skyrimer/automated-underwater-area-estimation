from typing import Dict, ClassVar
from dataclasses import dataclass
from automated_underwater_area_estimation.segmentation.class_mapping import (
    ClassMappingBase,
)


@dataclass
class ReefSupportClassMapping(ClassMappingBase):
    """Class mapping for the ReefSupport YOLO coral segmentation models."""

    # Based on typical coral reef segmentation classes - you may need to adjust these
    CLASS_NAMES: ClassVar[Dict[int, str]] = {
        0: "background",
        1: "hard coral",
        2: "soft coral",
    }
