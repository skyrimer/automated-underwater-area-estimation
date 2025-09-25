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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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
        """Segment an image using the EPFL model."""
        if adjust_size:
            image = image.resize(self.ideal_size)

        inputs = self.preprocess(image)
        with torch.no_grad():
            outputs = self.model(**inputs)

        segments = self.preprocessor.post_process_semantic_segmentation(
            outputs, target_sizes=[(image.size[1], image.size[0])]
        )[0]

        return image, segments

    def preprocess(self, image: Image.Image) -> BatchFeature:
        """Preprocess image for EPFL model input."""
        return self.preprocessor(image, return_tensors="pt").to(self.device)

    def visualize_segmentation(
        self,
        image: Image.Image,
        segments: torch.Tensor = None,
        alpha: float = 0.6,
        figsize: Tuple[int, int] = (15, 10),
        show_legend: bool = True,
        adjust_size: bool = True,
    ) -> plt.Figure:
        """
        Visualize segmentation results with original image and overlaid segments.

        Args:
            image: Original PIL Image
            segments: Optional pre-computed segmentation tensor. If None, will compute segments.
            alpha: Transparency for segment overlay (0.0 = transparent, 1.0 = opaque)
            figsize: Figure size as (width, height)
            show_legend: Whether to show class legend
            adjust_size: Whether to adjust image size before segmentation

        Returns:
            matplotlib Figure object
        """
        # Get segments if not provided
        if segments is None:
            processed_image, segments = self.segment_image(
                image, adjust_size=adjust_size
            )
        else:
            processed_image = image.resize(self.ideal_size) if adjust_size else image

        # Convert tensor to numpy array
        segments_np = segments.cpu().numpy()

        # Convert PIL image to numpy array
        image_np = np.array(processed_image)

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Plot 1: Original image
        axes[0].imshow(image_np)
        axes[0].set_title("Original Image", fontsize=14)
        axes[0].axis("off")

        # Plot 2: Segmentation mask
        # Create a colormap for all classes
        unique_classes = np.unique(segments_np)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))

        # Create colored segmentation mask
        colored_mask = np.zeros((*segments_np.shape, 3))
        for i, class_id in enumerate(unique_classes):
            mask = segments_np == class_id
            colored_mask[mask] = colors[i][:3]

        axes[1].imshow(colored_mask)
        axes[1].set_title("Segmentation Mask", fontsize=14)
        axes[1].axis("off")

        # Plot 3: Overlay
        axes[2].imshow(image_np)
        axes[2].imshow(colored_mask, alpha=alpha)
        axes[2].set_title("Overlay", fontsize=14)
        axes[2].axis("off")

        # Add legend if requested
        if show_legend:
            legend_patches = []
            for i, class_id in enumerate(unique_classes):
                class_name = self.class_mapping.get_name(int(class_id))
                color = colors[i][:3]
                patch = mpatches.Patch(color=color, label=f"{class_id}: {class_name}")
                legend_patches.append(patch)

            # Add legend outside the plots
            fig.legend(
                handles=legend_patches,
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
                fontsize=10,
            )

        plt.tight_layout()
        return fig
