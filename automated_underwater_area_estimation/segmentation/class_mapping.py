from abc import ABC, abstractmethod
from typing import Dict, List, ClassVar
from dataclasses import dataclass


@dataclass
class ClassMappingBase(ABC):
    """
    Abstract base class for class mappings in segmentation datasets.
    Provides validation and common interface for all class mapping implementations.
    """

    # Abstract class variable that must be defined in subclasses
    CLASS_NAMES: ClassVar[Dict[int, str]]

    def __post_init__(self):
        """Validate the class mapping after initialization."""
        self._validate_class_mapping()

    def _validate_class_mapping(self) -> None:
        """Validate that the class mapping follows required constraints."""
        if not hasattr(self, "CLASS_NAMES") or not self.CLASS_NAMES:
            raise ValueError("CLASS_NAMES must be defined and non-empty")

        # Check that all class names are strings and non-empty
        for class_id, class_name in self.CLASS_NAMES.items():
            if not isinstance(class_name, str) or not class_name.strip():
                raise ValueError(
                    f"Class name for ID {class_id} must be a non-empty string, "
                    f"got: {class_name}"
                )

        # Check for duplicate class names
        class_names = list(self.CLASS_NAMES.values())
        if len(class_names) != len(set(class_names)):
            duplicates = {name for name in class_names if class_names.count(name) > 1}
            raise ValueError(f"Duplicate class names found: {duplicates}")

    @classmethod
    def get_name(cls, class_id: int) -> str:
        """Get class name by ID."""
        return cls.CLASS_NAMES.get(class_id, f"unknown_class_{class_id}")

    @classmethod
    def get_id(cls, class_name: str) -> int:
        """Get class ID by name."""
        for id_, name in cls.CLASS_NAMES.items():
            if name == class_name:
                return id_
        raise ValueError(f"Class name '{class_name}' not found")

    @classmethod
    def get_all_names(cls) -> List[str]:
        """Get all class names in order of their IDs."""
        return [cls.CLASS_NAMES[i] for i in sorted(cls.CLASS_NAMES.keys())]

    @classmethod
    def get_num_classes(cls) -> int:
        """Get total number of classes."""
        return len(cls.CLASS_NAMES)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(dataset={self.get_dataset_name()}, classes={self.get_num_classes()})"

    def __repr__(self) -> str:
        return self.__str__()
