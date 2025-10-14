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
# ANGLES: Sequence[int | float] = [0, 15, -15, 30, -30, 45, -45]
ANGLES: Sequence[int | float] = [0, 15, -15]
EXPAND = False  # rotation is handled via Albumentations w/ border fill

# How many randomized augmentations to generate per fixed angle
AUGS_PER_ANGLE = 2  # e.g., 2 variants for each angle

# Output size (downscale from 4000x3000 to something manageable for training)
# If None, keep original size; otherwise (W, H)
OUTPUT_SIZE: Optional[Tuple[int, int]] = (800, 600)

# Set a non-black border for visibility or keep black
BORDER_VALUE_IMG = (0, 0, 0)
BORDER_VALUE_MASK = 0


# ----------------------------
# HELPERS
# ----------------------------
def load_mask_pt(path: str) -> torch.Tensor:
    """
    Load a binary mask saved as a .pt (torch tensor or dict with 'mask').
    Returns torch.bool of shape (H, W).
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


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    """PIL RGB -> OpenCV BGR uint8"""
    arr = np.asarray(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """OpenCV BGR -> PIL RGB"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def mask_to_cv2(mask: torch.Tensor) -> np.ndarray:
    """
    torch.bool (H,W) -> uint8 mask with explicit channel (H,W,1) for Albumentations.
    Values are {0,255}.
    """
    m = mask.detach().cpu().numpy().astype(np.uint8) * 255
    if m.ndim == 2:
        m = m[..., None]  # (H,W) -> (H,W,1)
    return m


def cv2_to_mask(mask_u8: np.ndarray) -> torch.Tensor:
    """
    (H,W) or (H,W,1) uint8 -> torch.bool (H,W).
    """
    if mask_u8.ndim == 3 and mask_u8.shape[-1] == 1:
        mask_u8 = mask_u8[..., 0]
    return torch.from_numpy((mask_u8 > 127).astype(np.bool_))


# ----------------------------
# AUGMENTATION PIPELINES
# ----------------------------
def make_rotation_only(angle: float) -> A.BasicTransform:
    # Exact rotation to preserve your legacy “fixed angles”
    return A.Rotate(
        limit=(angle, angle),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        value=BORDER_VALUE_IMG,
        mask_value=BORDER_VALUE_MASK,
        always_apply=True,
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

    transforms = [
        # Small random affine jitter around the fixed angle rotation
        A.ShiftScaleRotate(
            shift_limit=0.05,  # +/- 5% translate
            scale_limit=0.15,  # +/- 15% scale
            rotate_limit=3,  # tiny extra rotation jitter
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            value=BORDER_VALUE_IMG,
            mask_value=BORDER_VALUE_MASK,
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
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
        # Random occlusions (simulate sea creatures/algae covering edges)
        A.CoarseDropout(
            max_holes=6,
            max_height=0.1,
            max_width=0.1,
            min_holes=1,
            min_height=0.03,
            min_width=0.03,
            fill_value=BORDER_VALUE_IMG,
            mask_fill_value=0,
            p=0.35,
        ),
        # Cropping strategy:
        # mostly center crop (quadrant near center), sometimes random resized crop
        # Calculate actual pixel dimensions if image_size is provided
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


def main():
    img_dir = Path(IMG_DIR)
    mask_dir = Path(MASK_DIR)
    out_img_dir = Path(OUT_IMG_DIR)
    out_mask_dir = Path(OUT_MASK_DIR)

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    img_files = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    img_files.sort()

    if not img_files:
        print(f"No images found in {IMG_DIR}.")
        return

    print(f"Found {len(img_files)} images.")
    print(
        f"Fixed rotations: {ANGLES} | Augs per angle: {AUGS_PER_ANGLE} | Output size: {OUTPUT_SIZE}"
    )

    # Pre-build per-angle fixed rotation transforms (albumentations)
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

        # Load image & mask
        pil_img = Image.open(img_path).convert("RGB")
        mask_t = load_mask_pt(str(mask_path))  # torch.bool (H, W)

        # Convert to albumentations-friendly formats
        img_cv = pil_to_cv2(pil_img)
        mask_cv = mask_to_cv2(mask_t)  # uint8 {0,255}

        for angle in ANGLES:
            # 1) Deterministic rotation (exactly your chosen angle)
            out1 = rot_xforms[float(angle)](image=img_cv, mask=mask_cv)
            base_img = out1["image"]
            base_mask = out1["mask"]

            # 2) Now produce N randomized variants per angle
            for k in range(AUGS_PER_ANGLE):
                out2 = rand_xform(image=base_img, mask=base_mask)
                aug_img = out2["image"]
                aug_mask = out2["mask"]

                # Convert back to PIL / torch.bool
                pil_out = cv2_to_pil(aug_img)
                mask_out = cv2_to_mask(aug_mask)

                angle_tag = f"rot{int(angle):+03d}".replace("+", "")
                out_img_name = f"{stem}_{angle_tag}_aug{k:02d}.jpg"
                out_mask_name = f"{stem}_{angle_tag}_aug{k:02d}.pt"

                if pil_out.mode != "RGB":
                    pil_out = pil_out.convert("RGB")

                pil_out.save(out_img_dir / out_img_name, quality=95, subsampling=1)
                torch.save(mask_out, out_mask_dir / out_mask_name)

        print(f"Augmented {img_path.name} -> {len(ANGLES) * AUGS_PER_ANGLE} samples")

    print("Done.")


if __name__ == "__main__":
    main()
