from pathlib import Path
from typing import Tuple, Optional, Sequence

import numpy as np
from PIL import Image
import torch

import albumentations as A
import cv2
from tqdm.auto import tqdm

# ----------------------------
# CONFIG
# ----------------------------
base = "./automated_underwater_area_estimation/data_preprocessed/IBF/"
IMG_DIR = base + "images"
MASK_DIR = base + "improved_masks"
OUT_IMG_DIR = base + "out_images"
OUT_MASK_DIR = base + "out_masks"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Keep your fixed angles, but we’ll also apply richer random augs per angle
ANGLES: Sequence[int | float] = [0, 15, -15]
EXPAND = False  # rotation is handled via Albumentations w/ border fill

# How many randomized augmentations to generate per fixed angle
AUGS_PER_ANGLE = 5

# Output size (downscale from 4000x3000 to something manageable for training)
OUTPUT_SIZE: Optional[Tuple[int, int]] = (800, 600)

# Set a non-black border for visibility or keep black
BORDER_VALUE_IMG = (0, 0, 0)
BORDER_VALUE_MASK = 0


# ----------------------------
# HELPERS
# ----------------------------
def load_mask_pt(path: str) -> torch.Tensor:
    """
    Load a binary mask saved as a .pt file.
    
    Handles masks stored as torch tensors or dictionaries with a 'mask' key.
    Converts the loaded data to a boolean tensor of shape (H, W).
    
    Args:
        path: Path to the .pt file containing the mask
        
    Returns:
        Boolean torch.Tensor of shape (H, W)
        
    Raises:
        ValueError: If the mask format is unsupported
    """
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict) and "mask" in data:
        t = data["mask"]
    else:
        t = data
    if isinstance(t, torch.Tensor):
        # Accept float/bool/long; convert to bool
        if t.ndim == 3 and t.shape[0] in (1, 3):  # (C,H,W) -> (H,W)
            t = t.squeeze(0)
        t = t > 0.5 if t.dtype.is_floating_point else t.bool()
        return t
    raise ValueError(f"Unsupported mask format in {path}: {type(t)}")


def mask_to_cv2(mask: torch.Tensor) -> np.ndarray:
    """
    Convert torch boolean mask to OpenCV-compatible uint8 format.
    
    Converts a torch boolean tensor to uint8 numpy array with shape (H, W, 1)
    for use with Albumentations. Values are {0, 255}.
    
    Args:
        mask: Boolean torch.Tensor of shape (H, W)
        
    Returns:
        Uint8 numpy array of shape (H, W, 1) with values in {0, 255}
    """
    m = mask.detach().cpu().numpy().astype(np.uint8) * 255
    if m.ndim == 2:
        m = m[..., None]  # (H,W) -> (H,W,1)
    return m


def cv2_to_mask(mask_u8: np.ndarray) -> torch.Tensor:
    """
    Convert OpenCV uint8 mask back to torch boolean tensor.
    
    Converts a uint8 numpy array of shape (H, W) or (H, W, 1) to a
    boolean torch.Tensor of shape (H, W).
    
    Args:
        mask_u8: Uint8 numpy array of shape (H, W) or (H, W, 1)
        
    Returns:
        Boolean torch.Tensor of shape (H, W)
    """
    if mask_u8.ndim == 3 and mask_u8.shape[-1] == 1:
        mask_u8 = mask_u8[..., 0]
    return torch.from_numpy((mask_u8 > 127).astype(np.bool_))


# ----------------------------
# AUGMENTATION PIPELINES
# ----------------------------
def make_rotation_only(angle: float) -> A.BasicTransform:
    """
    Create an Albumentations transform for exact rotation at a fixed angle.
    
    Args:
        angle: Rotation angle in degrees
        
    Returns:
        Albumentations Rotate transform configured for the specified angle
    """
    # Exact rotation to preserve your legacy "fixed angles"
    return A.Rotate(
        limit=(angle, angle),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        fill=BORDER_VALUE_IMG,
        fill_mask=BORDER_VALUE_MASK,
        p=1,
    )


def make_random_pipeline(
    image_size: Optional[Tuple[int, int]] = OUTPUT_SIZE,
) -> A.Compose:
    """
    Randomized augmentations tailored for underwater imagery and partial occlusion.
    Order is important: geom -> color -> blur/noise -> occlusion -> resize/crop.
    """
    # If the quadrant is usually near center: bias crops to center most of the time.
    center_crop_prob = 0
    scale_factor = 0.1
    transforms = [
        # Small random affine jitter around the fixed angle rotation
        A.Affine(
            translate_percent={"x": 0.05, "y": 0.05},
            scale=(1 - scale_factor, 1 + scale_factor),
            rotate=(-3, 3),
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            fill=BORDER_VALUE_IMG,
            fill_mask=BORDER_VALUE_MASK,
            p=0.9,
        ),
        A.HorizontalFlip(p=0.5),
        # Color / lighting jitter (underwater lighting variability)
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.9),
        A.HueSaturationValue(
            hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=12, p=0.7
        ),
        # Blur / noise (turbidity, motion blur-like effects)
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ],
            p=0.25,
        ),
        A.GaussNoise(std_range=(0.05, 0.2), p=0.3),
        # Random occlusions (simulate sea creatures/algae covering edges)
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(int(0.03 * image_size[1]), int(0.10 * image_size[1])),
            hole_width_range=(int(0.03 * image_size[0]), int(0.10 * image_size[0])),
            fill=BORDER_VALUE_IMG,
            fill_mask=0,
            p=0.35,
        ),
    ]

    if center_crop_prob > 0 and image_size is not None:
        # Calculate 85% and 90% crop sizes in pixels
        crop_h_85 = int(image_size[1] * 0.85)
        crop_w_85 = int(image_size[0] * 0.85)
        crop_h_90 = int(image_size[1] * 0.9)
        crop_w_90 = int(image_size[0] * 0.9)

        transforms.append(
            A.OneOf(
                [
                    A.CenterCrop(height=crop_h_85, width=crop_w_85, p=1.0),
                    A.RandomResizedCrop(
                        height=crop_h_90,
                        width=crop_w_90,
                        scale=(0.75, 1.15),
                        ratio=(0.95, 1.05),
                        interpolation=cv2.INTER_LINEAR,
                        p=1.0,
                    ),
                ],
                p=center_crop_prob,
            )
        )

    # Ensure final size
    # If OUTPUT_SIZE is None, skip; else resize to that fixed train size
    if image_size is not None:
        transforms.append(
            A.Resize(
                height=image_size[1],
                width=image_size[0],
                interpolation=cv2.INTER_LINEAR,
                p=1.0,
            )
        )
    else:
        transforms.append(A.NoOp())

    return A.Compose(
        transforms,
        additional_targets={},  # we only have one image+mask
        is_check_shapes=False,
    )


def main() -> None:
    """
    Main function to perform data augmentation on images and masks.
    
    Applies fixed rotations and random augmentations to create an augmented dataset
    suitable for training segmentation models. Reads images and masks from configured
    directories and saves augmented versions to output directories.
    """
    img_dir = Path(IMG_DIR)
    mask_dir = Path(MASK_DIR)
    out_img_dir = Path(OUT_IMG_DIR)
    out_mask_dir = Path(OUT_MASK_DIR)

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    # List image files
    img_files = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    img_files.sort()

    if not img_files:
        print(f"No images found in {IMG_DIR}.")
        return

    print(f"Found {len(img_files)} images.")
    print(
        f"Fixed rotations: {ANGLES} | Augs per angle: {AUGS_PER_ANGLE} | Output size: {OUTPUT_SIZE}"
    )

    # Pre-build rotation and random transforms
    rot_xforms = {float(a): make_rotation_only(float(a)) for a in ANGLES}
    rand_xform = make_random_pipeline(OUTPUT_SIZE)

    for img_path in tqdm(img_files, desc="Augmenting images"):
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}_improved.pt"
        if not mask_path.exists():
            print(
                f"[WARN] Missing mask for image {img_path.name} -> expected {mask_path.name}. Skipping."
            )
            continue

        # Load image as BGR uint8 via OpenCV
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[ERROR] Could not read image {img_path}. Skipping.")
            continue

        # Load the mask from .pt
        mask_t = load_mask_pt(str(mask_path))  # torch.bool (H, W)
        # Convert mask to uint8 with {0,255}, with shape (H, W, 1)
        mask_np = mask_to_cv2(mask_t)

        # --- AUGMENT & SAVE ---
        for angle in ANGLES:
            out1 = rot_xforms[float(angle)](image=img_bgr, mask=mask_np)
            base_img = out1["image"]
            base_mask = out1["mask"]

            for k in range(AUGS_PER_ANGLE):
                out2 = rand_xform(image=base_img, mask=base_mask)
                aug_img = out2["image"]
                aug_mask = out2["mask"]

                # Convert aug_img (BGR) to PIL for saving (RGB)
                rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
                pil_out = Image.fromarray(rgb)

                # Convert mask for saving
                mask_out = cv2_to_mask(aug_mask)

                # Build filenames
                angle_tag = f"rot{int(angle):+03d}".replace("+", "")
                out_img_name = f"{stem}_{angle_tag}_aug{k:02d}.jpg"
                out_mask_name = f"{stem}_{angle_tag}_aug{k:02d}.pt"

                # Save image — using PIL (you could also use cv2.imwrite)
                if pil_out.mode != "RGB":
                    pil_out = pil_out.convert("RGB")
                pil_out.save(out_img_dir / out_img_name, quality=95, subsampling=1)

                # Save mask tensor
                torch.save(mask_out, out_mask_dir / out_mask_name)


if __name__ == "__main__":
    main()
