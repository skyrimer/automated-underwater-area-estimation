import pytest
import torch
from PIL import Image
from datasets import load_dataset
from typing import Tuple

from automated_underwater_area_estimation.segmentation.epfl.model import EPFLModel
from automated_underwater_area_estimation.segmentation.reefsupport.model import (
    ReefSupportModel,
)


class TestSegmentationModels:
    """Test suite for segmentation models."""

    @classmethod
    def setup_class(cls):
        """Set up test fixtures that will be shared across all tests."""
        # Load a sample image from CoralScapes dataset
        print("Loading CoralScapes dataset...")
        dataset = load_dataset("EPFL-ECEO/coralscapes", split="test")
        cls.test_image = dataset[42][
            "image"
        ]  # Use a specific index for reproducibility

        # Ensure it's a PIL Image
        if not isinstance(cls.test_image, Image.Image):
            cls.test_image = Image.fromarray(cls.test_image)

    def test_epfl_model_loading_b2(self):
        """Test that EPFL B2 model loads successfully."""
        model = EPFLModel("EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024")

        # Check required attributes
        assert hasattr(model, "model_name")
        assert hasattr(model, "preprocessor")
        assert hasattr(model, "model")
        assert hasattr(model, "class_mapping")
        assert hasattr(model, "ideal_size")
        assert hasattr(model, "device")

        # Check specific values
        assert (
            model.model_name == "EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024"
        )
        assert model.ideal_size == (1024, 1024)
        assert isinstance(model.device, torch.device)

    def test_epfl_model_loading_b5(self):
        """Test that EPFL B5 model loads successfully."""
        model = EPFLModel("EPFL-ECEO/segformer-b5-finetuned-coralscapes-1024-1024")

        # Check required attributes
        assert hasattr(model, "model_name")
        assert hasattr(model, "preprocessor")
        assert hasattr(model, "model")
        assert hasattr(model, "class_mapping")
        assert hasattr(model, "ideal_size")
        assert hasattr(model, "device")

        # Check specific values
        assert (
            model.model_name == "EPFL-ECEO/segformer-b5-finetuned-coralscapes-1024-1024"
        )
        assert model.ideal_size == (1024, 1024)
        assert isinstance(model.device, torch.device)

    def test_reefsupport_model_loading_sm(self):
        """Test that ReefSupport small model loads successfully."""
        model = ReefSupportModel("yolov8_sm_latest.pt")

        # Check required attributes
        assert hasattr(model, "model_name")
        assert hasattr(model, "preprocessor")
        assert hasattr(model, "model")
        assert hasattr(model, "class_mapping")
        assert hasattr(model, "ideal_size")
        assert hasattr(model, "device")

        # Check specific values
        assert model.model_name == "yolov8_sm_latest.pt"
        assert model.ideal_size == (640, 640)
        assert isinstance(model.device, torch.device)
        assert model.preprocessor is None  # YOLO handles preprocessing internally

    def test_reefsupport_model_loading_xlarge(self):
        """Test that ReefSupport xlarge model loads successfully."""
        model = ReefSupportModel("yolov8_xlarge_latest.pt")

        # Check required attributes
        assert hasattr(model, "model_name")
        assert hasattr(model, "preprocessor")
        assert hasattr(model, "model")
        assert hasattr(model, "class_mapping")
        assert hasattr(model, "ideal_size")
        assert hasattr(model, "device")

        # Check specific values
        assert model.model_name == "yolov8_xlarge_latest.pt"
        assert model.ideal_size == (640, 640)
        assert isinstance(model.device, torch.device)
        assert model.preprocessor is None

    def test_epfl_model_segmentation_b2(self):
        """Test EPFL B2 model segmentation on CoralScapes image."""
        model = EPFLModel("EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024")

        # Perform segmentation
        result = model.segment_image(self.test_image, adjust_size=True)

        # Check return type
        assert isinstance(result, tuple)
        assert len(result) == 2

        processed_image, segmentation_mask = result

        # Check processed image
        assert isinstance(processed_image, Image.Image)
        assert processed_image.size == (1024, 1024)  # Should be resized to ideal size

        # Check segmentation mask
        assert isinstance(segmentation_mask, torch.Tensor)
        assert segmentation_mask.dtype in [torch.long, torch.int64]
        assert len(segmentation_mask.shape) == 2  # Should be 2D

        # Check that mask has reasonable values (class IDs should be non-negative)
        assert segmentation_mask.min() >= 0

        # Check that mask dimensions match expected size
        assert segmentation_mask.shape[0] > 0 and segmentation_mask.shape[1] > 0

    def test_epfl_model_segmentation_b5(self):
        """Test EPFL B5 model segmentation on CoralScapes image."""
        model = EPFLModel("EPFL-ECEO/segformer-b5-finetuned-coralscapes-1024-1024")

        # Perform segmentation
        result = model.segment_image(self.test_image, adjust_size=True)

        # Check return type
        assert isinstance(result, tuple)
        assert len(result) == 2

        processed_image, segmentation_mask = result

        # Check processed image
        assert isinstance(processed_image, Image.Image)
        assert processed_image.size == (1024, 1024)

        # Check segmentation mask
        assert isinstance(segmentation_mask, torch.Tensor)
        assert segmentation_mask.dtype in [torch.long, torch.int64]
        assert len(segmentation_mask.shape) == 2
        assert segmentation_mask.min() >= 0

    def test_reefsupport_model_segmentation_sm(self):
        """Test ReefSupport small model segmentation on CoralScapes image."""
        model = ReefSupportModel("yolov8_sm_latest.pt")

        # Perform segmentation
        result = model.segment_image(self.test_image, adjust_size=True)

        # Check return type
        assert isinstance(result, tuple)
        assert len(result) == 2

        processed_image, segmentation_mask = result

        # Check processed image
        assert isinstance(processed_image, Image.Image)
        assert processed_image.size == (
            640,
            640,
        )  # Should be resized to YOLO ideal size

        # Check segmentation mask
        assert isinstance(segmentation_mask, torch.Tensor)
        assert segmentation_mask.dtype in [torch.long, torch.int64]
        assert len(segmentation_mask.shape) == 2
        assert segmentation_mask.min() >= 0

    def test_reefsupport_model_segmentation_xlarge(self):
        """Test ReefSupport xlarge model segmentation on CoralScapes image."""
        model = ReefSupportModel("yolov8_xlarge_latest.pt")

        # Perform segmentation
        result = model.segment_image(self.test_image, adjust_size=True)

        # Check return type
        assert isinstance(result, tuple)
        assert len(result) == 2

        processed_image, segmentation_mask = result

        # Check processed image
        assert isinstance(processed_image, Image.Image)
        assert processed_image.size == (640, 640)

        # Check segmentation mask
        assert isinstance(segmentation_mask, torch.Tensor)
        assert segmentation_mask.dtype in [torch.long, torch.int64]
        assert len(segmentation_mask.shape) == 2
        assert segmentation_mask.min() >= 0

    def test_device_specification(self):
        """Test that models respect device specification."""
        # Test with CPU device
        cpu_device = torch.device("cpu")
        model = EPFLModel(
            "EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024", device=cpu_device
        )

        assert model.device == cpu_device

        # Check that model parameters are on the correct device
        for param in model.model.parameters():
            assert param.device == cpu_device
            break  # Just check the first parameter

    def test_preprocessing_methods(self):
        """Test preprocessing methods work correctly."""
        epfl_model = EPFLModel("EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024")
        reef_model = ReefSupportModel("yolov8_sm_latest.pt")

        # Test EPFL preprocessing
        epfl_preprocessed = epfl_model.preprocess(self.test_image)
        assert epfl_preprocessed is not None
        # Should return a BatchFeature that can be unpacked with **

        # Test ReefSupport preprocessing
        reef_preprocessed = reef_model.preprocess(self.test_image)
        assert reef_preprocessed is not None
        # Should return numpy array
        import numpy as np

        assert isinstance(reef_preprocessed, np.ndarray)

    def test_class_mapping_methods(self):
        """Test class mapping functionality."""
        model = EPFLModel("EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024")

        # Test class mapping methods exist and work
        num_classes = model.get_num_classes()
        assert isinstance(num_classes, int)
        assert num_classes > 0

        all_class_names = model.get_all_class_names()
        assert isinstance(all_class_names, list)
        assert len(all_class_names) == num_classes

        # Test getting individual class names
        if num_classes > 0:
            class_name = model.get_class_name(0)
            assert isinstance(class_name, str)

    def test_model_string_representation(self):
        """Test string representation of models."""
        epfl_model = EPFLModel("EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024")
        reef_model = ReefSupportModel("yolov8_sm_latest.pt")

        # Test __str__ method
        epfl_str = str(epfl_model)
        reef_str = str(reef_model)

        assert "EPFLModel" in epfl_str
        assert "ReefSupportModel" in reef_str
        assert "EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024" in epfl_str
        assert "yolov8_sm_latest.pt" in reef_str

        # Test __repr__ method
        assert repr(epfl_model) == str(epfl_model)
        assert repr(reef_model) == str(reef_model)

    def test_invalid_model_names(self):
        """Test that invalid model names raise appropriate errors."""
        with pytest.raises(ValueError):
            EPFLModel("invalid-model-name")

        with pytest.raises(ValueError):
            ReefSupportModel("invalid-model.pt")

    @pytest.mark.parametrize("adjust_size", [True, False])
    def test_segmentation_with_size_adjustment(self, adjust_size):
        """Test segmentation with and without size adjustment."""
        model = EPFLModel("EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024")

        result = model.segment_image(self.test_image, adjust_size=adjust_size)
        processed_image, segmentation_mask = result

        assert isinstance(processed_image, Image.Image)
        assert isinstance(segmentation_mask, torch.Tensor)

        if adjust_size:
            assert processed_image.size == (1024, 1024)
        # If not adjusting size, the image should keep its original dimensions
