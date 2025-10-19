import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from automated_underwater_area_estimation.utils import get_best_device
from pathlib import Path


class QuadrantSegmentationModel:
    def __init__(
        self,
        device: str = None,
        target_size: tuple[int, int] = (800, 600),
    ):
        """
        checkpoint_dir: path to directory (or HF model name) where the model+processor are stored
        device: e.g., "cuda" or "cpu". If None it will auto‐detect.
        target_size: optional (width, height) tuple to resize input images before processing (same as used in training)
        """
        device = device or get_best_device()
        self.device = torch.device(device)
        checkpoint_dir = Path(__file__).parent / "segformer_best"
        assert  checkpoint_dir.exists(), "Model checkpoint not found. Check that you have loaded the weights"
        self.processor = AutoImageProcessor.from_pretrained(checkpoint_dir)
        self.model = (
            SegformerForSemanticSegmentation.from_pretrained(checkpoint_dir)
            .to(self.device)
            .eval()
        )
        self.target_size = target_size  # e.g., (800,600) width/height

    def segment_image(self, image: Image.Image) -> np.ndarray:
        """
        Segment a single PIL image and return a 2D numpy array mask of class labels (H_orig × W_orig),
        with values like {0,1,2,...} per pixel.
        """
        # resize / preprocess
        if self.target_size is not None:
            width, height = self.target_size
            encoded = self.processor(
                images=image,
                size={"height": height, "width": width},
                return_tensors="pt",
            )
        else:
            encoded = self.processor(images=image, return_tensors="pt")

        # move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)

        # post-process (resize back to original)
        orig_size = image.size  # (width, height)
        seg_maps = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(orig_size[1], orig_size[0])]
        )
        mask = seg_maps[0].cpu().numpy().astype(np.uint8)  # shape (H_orig, W_orig)
        return mask

    @staticmethod
    def overlay_mask(
        image: Image.Image,
        mask: np.ndarray,
        colour: tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.5,
    ) -> Image.Image:
        """
        Overlay the mask onto the original image and return a PIL image.
        - image: PIL RGB
        - mask: 2D numpy array same size as image; mask==label_to_overlay will be coloured.
        - colour: RGB tuple for overlay
        - alpha: blending factor (0.0 → just image, 1.0 → full colour)
        """
        img_np = np.array(image.convert("RGB"))
        # ensure mask matches size
        if mask.shape[0] != img_np.shape[0] or mask.shape[1] != img_np.shape[1]:
            mask_pil = Image.fromarray(mask.astype("uint8") * 255)
            mask_pil = mask_pil.resize(image.size, resample=Image.NEAREST)
            mask = np.array(mask_pil) // 255

        overlay_np = img_np.copy()
        colour_arr = np.array(colour, dtype=np.uint8)
        # apply overlay where mask ≠ 0 (assuming 0 is background)
        overlay_region = mask != 0
        overlay_np[overlay_region] = (
            overlay_np[overlay_region] * (1 - alpha) + colour_arr * alpha
        ).astype(np.uint8)

        return Image.fromarray(overlay_np)

    def segment_and_overlay(
        self,
        image_path: str,
        output_mask_path: str = None,
        output_overlay_path: str = None,
        colour: tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.5,
    ) -> tuple[np.ndarray, Image.Image]:
        """
        Convenience wrapper: loads image from path, performs segmentation, optionally saves mask and overlay.
        Returns: (mask array, overlay PIL image)
        """
        image = Image.open(image_path).convert("RGB")
        mask = self.segment_image(image)
        overlay = self.overlay_mask(image, mask, colour=colour, alpha=alpha)

        if output_mask_path:
            # save mask as uint8 image (0/255)
            mask_img = Image.fromarray((mask.astype("uint8") * 255))
            mask_img.save(output_mask_path)

        if output_overlay_path:
            overlay.save(output_overlay_path)

        return mask, overlay
