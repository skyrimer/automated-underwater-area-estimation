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
        self.model: YOLO = YOLO(self.model_path)

        self.preprocessor: None = None  # YOLO handles preprocessing internally
        self.class_mapping: ReefSupportClassMapping = ReefSupportClassMapping()
        self.ideal_size: Tuple[int, int] = (640, 640)  # Standard YOLO input size
        self.device: torch.device = device or get_best_device()
        # Call parent validation
        super().__init__(self.device)

    def segment_image(
        self, image: Image.Image, adjust_size: bool = True
    ) -> Tuple[Image.Image, torch.Tensor]:
        """Segment an image using the ReefSupport YOLO model."""
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
                # Combine all masks into a single segmentation map
                # Each mask gets the class ID of its detection
                image_height, image_width = image.size
                segmentation_map = torch.zeros(
                    (image_height, image_width), dtype=torch.long
                )

                for i, mask in enumerate(masks):
                    # Get class ID for this detection
                    class_id = (
                        int(results[0].boxes.cls[i].item()) + 1
                    )  # +1 because our mapping starts at 1
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
                    segmentation_map[mask_resized] = class_id

                return image, segmentation_map

        # Return empty segmentation if no masks found
        empty_segmentation = torch.zeros((image.height, image.width), dtype=torch.long)
        return image, empty_segmentation

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
