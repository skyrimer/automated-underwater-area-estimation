from pathlib import Path
from typing import Tuple, List, Optional, Dict
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
from PIL import Image
import torch

import matplotlib

matplotlib.use("Agg")  # ensure non-interactive backend
import matplotlib.pyplot as plt

from skimage import morphology, measure
from tqdm.auto import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def get_disk_footprint(radius: int) -> np.ndarray:
    """Get or create disk footprint (cache is per-process in multiprocessing)."""
    return morphology.disk(radius)


def load_image(path: Path) -> np.ndarray:
    """Load image as RGB uint8 numpy array."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def load_mask_pt(path: Path, expected_shape: Tuple[int, int]) -> np.ndarray:
    """Load a binary mask saved as a .pt tensor and return bool numpy array with shape (H, W)."""
    data = torch.load(path, map_location="cpu", weights_only=False)

    # Accept torch.Tensor or dict with 'mask' key
    t = data["mask"] if isinstance(data, dict) and "mask" in data else data

    if isinstance(t, torch.Tensor):
        arr = t.detach().cpu().numpy()
    else:
        arr = np.asarray(t)

    # Squeeze and convert to boolean in one step
    arr = np.squeeze(arr)

    # Handle shape validation
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape != expected_shape:
        raise ValueError(
            f"Mask shape {arr.shape} does not match image shape {expected_shape} for {path.name}"
        )

    # Convert to boolean (handles all numeric types efficiently)
    return arr.astype(bool) if arr.dtype != bool else arr


def keep_largest_component(mask: np.ndarray, connectivity: int = 2) -> np.ndarray:
    """Keep only the largest connected component in a boolean mask."""
    if not mask.any():
        return mask

    # Use return_num for more efficient labeling
    labeled, num_labels = measure.label(
        mask, connectivity=connectivity, return_num=True
    )

    if num_labels == 0:
        return mask
    if num_labels == 1:
        return mask  # Already single component

    # More efficient: use bincount instead of unique
    counts = np.bincount(labeled.flat)[1:]  # Skip background
    largest_label = np.argmax(counts) + 1  # +1 because we skipped 0

    return labeled == largest_label


def opening_by_reconstruction(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary opening-by-reconstruction with a disk footprint."""
    if radius <= 0:
        return mask

    fp = get_disk_footprint(radius)
    seed = morphology.erosion(mask, footprint=fp)
    opened = morphology.reconstruction(seed, mask, method="dilation")
    return opened.astype(bool)


def closing_by_reconstruction(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary closing-by-reconstruction implemented via complement trick."""
    if radius <= 0:
        return mask

    fp = get_disk_footprint(radius)
    comp = ~mask
    seed = morphology.erosion(comp, footprint=fp)
    rec = morphology.reconstruction(seed, comp, method="dilation")
    return ~rec.astype(bool)


def improve_mask_skimage(
    mask: np.ndarray,
    image_shape: Tuple[int, int],
    min_obj_frac: float = 0.001,
    min_hole_frac: float = 0.003,
    r_open_frac: float = 0.004,
    r_close_frac: float = 0.006,
    r_recon_frac: float = 0.003,
    keep_largest: bool = True,
) -> np.ndarray:
    """
    Apply a robust skimage.morphology cleaning pipeline:
      1) (optional) keep largest connected component
      2) remove_small_objects
      3) remove_small_holes
      4) binary opening (light)
      5) opening by reconstruction (shape-preserving)
      6) binary closing (light)
      7) closing by reconstruction (gap filling, edge-preserving)
    """
    h, w = image_shape
    area = h * w
    min_obj_area = max(64, int(min_obj_frac * area))
    min_hole_area = max(64, int(min_hole_frac * area))

    # Radii scale with min(h, w)
    base = min(h, w)
    r_open = max(1, int(round(base * r_open_frac)))
    r_close = max(1, int(round(base * r_close_frac)))
    r_recon = max(1, int(round(base * r_recon_frac)))

    # Work with boolean mask (avoid unnecessary copy if already bool)
    m = mask if mask.dtype == bool else mask.astype(bool)

    if keep_largest:
        m = keep_largest_component(m, connectivity=2)

    # Remove speckles and pinholes
    m = morphology.remove_small_objects(m, min_size=min_obj_area, connectivity=2)
    m = morphology.remove_small_holes(m, area_threshold=min_hole_area, connectivity=2)

    # Light smoothing - use cached footprints
    m = morphology.binary_opening(m, footprint=get_disk_footprint(r_open))
    m = opening_by_reconstruction(m, radius=r_recon)

    m = morphology.binary_closing(m, footprint=get_disk_footprint(r_close))
    m = closing_by_reconstruction(m, radius=r_recon)

    # Final small-hole fill
    m = morphology.remove_small_holes(
        m, area_threshold=min_hole_area // 2, connectivity=2
    )

    return m


def overlay_side_by_side_and_save(
    image: np.ndarray,
    mask_left: np.ndarray,
    title_left: str,
    mask_right: np.ndarray,
    title_right: str,
    out_path: Path,
    dpi: int = 300,
) -> None:
    """
    Save a single high-resolution figure with two overlays side-by-side:
    left = original overlay, right = improved overlay.
    """
    h, w = image.shape[:2]

    # Dynamic figure size - compute once
    fig_width_in = max(12.0, 2.0 * w / 400.0)
    fig_height_in = max(6.0, h / 400.0)

    fig, axes = plt.subplots(1, 2, figsize=(fig_width_in, fig_height_in), dpi=dpi)

    for ax, m, t in [
        (axes[0], mask_left, title_left),
        (axes[1], mask_right, title_right),
    ]:
        ax.imshow(image)
        ax.imshow(np.ma.masked_where(~m, m), alpha=0.4)
        ax.set_axis_off()
        ax.set_title(t)

    fig.tight_layout(pad=0.25)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def process_single_image(
    img_path: Path,
    mask_map: Dict[str, Path],
    out_masks_dir: Path,
    out_plots_dir: Path,
    save_as_pt: bool,
    save_visualizations: bool,
    improve_kwargs: Dict,
) -> Optional[Tuple[str, Optional[Path], Optional[Path], Optional[Path]]]:
    """
    Process a single image/mask pair. This function is called by each worker process.

    Returns:
        Tuple of (basename, pt_out_path, png_out_path, plot_dir) or None if skipped/error
    """
    base = img_path.stem
    mask_path = mask_map.get(base)

    if mask_path is None:
        return None  # Will be filtered out later

    try:
        # Load mask first (cheaper operation) to get shape
        if save_visualizations:
            img = load_image(img_path)
            h, w = img.shape[:2]
        else:
            # For shape, we can peek at the mask
            mask_data = torch.load(mask_path, map_location="cpu", weights_only=False)
            temp_mask = mask_data["mask"] if isinstance(mask_data, dict) else mask_data
            if isinstance(temp_mask, torch.Tensor):
                h, w = temp_mask.shape[-2:]
            else:
                h, w = np.asarray(temp_mask).squeeze().shape[:2]
            img = None

        mask = load_mask_pt(mask_path, expected_shape=(h, w))
        improved = improve_mask_skimage(mask, (h, w), **improve_kwargs)

        # Save improved mask
        pt_out = None
        if save_as_pt:
            pt_out = out_masks_dir / f"{base}_improved.pt"
            torch.save(torch.from_numpy(improved.astype(np.uint8)), pt_out)

        # Save visualization only if requested
        png_out = None
        pair_plot_dir = None
        if save_visualizations:
            # Load image now if not already loaded
            if img is None:
                img = load_image(img_path)

            pair_plot_dir = out_plots_dir / base
            pair_plot_dir.mkdir(parents=True, exist_ok=True)

            side_by_side_path = pair_plot_dir / f"{base}_overlay_side_by_side.png"
            overlay_side_by_side_and_save(
                img,
                mask_left=mask,
                title_left=f"{base} — ORIGINAL",
                mask_right=improved,
                title_right=f"{base} — IMPROVED",
                out_path=side_by_side_path,
                dpi=300,
            )
            png_out = side_by_side_path

        return (base, pt_out, png_out, pair_plot_dir)

    except Exception as e:
        # Return error info as tuple
        return (base, None, None, None, str(e))


def process_dataset_parallel(
    images_dir: Path,
    masks_dir: Path,
    out_masks_dir: Path,
    out_plots_dir: Path,
    save_as_pt: bool = True,
    save_visualizations: bool = True,
    num_workers: Optional[int] = None,
    **improve_kwargs,
) -> List[Tuple[str, Optional[Path], Optional[Path], Optional[Path]]]:
    """
    Process all image/mask pairs using multiprocessing.

    Args:
        num_workers: Number of worker processes. If None, uses cpu_count() - 1

    Returns:
        List of tuples: (basename, improved_mask_pt_path, overlay_png_path, plot_dir)
    """
    # Create output directories
    out_masks_dir.mkdir(parents=True, exist_ok=True)
    if save_visualizations:
        out_plots_dir.mkdir(parents=True, exist_ok=True)

    # Build mapping from basename -> mask path (.pt)
    mask_map = {p.stem: p for p in masks_dir.glob("*.pt")}

    # Get all image files
    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

    if not images:
        print("[WARN] No images found to process.")
        return []

    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave 1 core free

    print(f"Processing {len(images)} images using {num_workers} workers...")

    # Create partial function with fixed arguments
    process_func = partial(
        process_single_image,
        mask_map=mask_map,
        out_masks_dir=out_masks_dir,
        out_plots_dir=out_plots_dir,
        save_as_pt=save_as_pt,
        save_visualizations=save_visualizations,
        improve_kwargs=improve_kwargs,
    )

    # Process in parallel with progress bar
    results = []
    errors = []
    skipped = []

    with Pool(processes=num_workers) as pool:
        # Use imap for better progress tracking
        for result in tqdm(
            pool.imap(process_func, images),
            total=len(images),
            desc="Processing images",
            unit="img",
        ):
            if result is None:
                # Image was skipped (no mask)
                skipped.append(None)
            elif len(result) == 5:
                # Error occurred
                base, _, _, _, error_msg = result
                errors.append((base, error_msg))
                print(f"[ERROR] {base}: {error_msg}")
            else:
                # Success
                results.append(result)
                if save_visualizations:
                    base, _, _, plot_dir = result
                    print(f"[OK] {base}: saved to {plot_dir}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Successfully processed: {len(results)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Skipped (no mask): {len(skipped)}")
    print(f"{'='*60}")

    if errors:
        print("\nErrors encountered:")
        for base, error in errors:
            print(f"  - {base}: {error}")

    return results


def process_dataset_sequential(
    images_dir: Path,
    masks_dir: Path,
    out_masks_dir: Path,
    out_plots_dir: Path,
    save_as_pt: bool = True,
    save_visualizations: bool = True,
    **improve_kwargs,
) -> List[Tuple[str, Optional[Path], Optional[Path], Optional[Path]]]:
    """
    Sequential processing (original implementation) - useful for debugging.
    """
    out_masks_dir.mkdir(parents=True, exist_ok=True)
    if save_visualizations:
        out_plots_dir.mkdir(parents=True, exist_ok=True)

    results = []
    mask_map = {p.stem: p for p in masks_dir.glob("*.pt")}
    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

    for img_path in tqdm(images, desc="Processing images"):
        result = process_single_image(
            img_path,
            mask_map,
            out_masks_dir,
            out_plots_dir,
            save_as_pt,
            save_visualizations,
            improve_kwargs,
        )

        if result is not None:
            if len(result) == 5:
                base, _, _, _, error = result
                print(f"[ERROR] {base}: {error}")
            else:
                results.append(result)
                if save_visualizations:
                    base, _, _, plot_dir = result
                    print(f"[OK] {base}: saved to {plot_dir}")

    return results


def process_dataset(
    images_dir: Path,
    masks_dir: Path,
    out_masks_dir: Path,
    out_plots_dir: Path,
    save_as_pt: bool = True,
    save_visualizations: bool = True,
    use_multiprocessing: bool = True,
    num_workers: Optional[int] = None,
    **improve_kwargs,
) -> List[Tuple[str, Optional[Path], Optional[Path], Optional[Path]]]:
    """
    Process all image/mask pairs with optional multiprocessing.

    Args:
        use_multiprocessing: If True, use parallel processing. If False, use sequential.
        num_workers: Number of worker processes (only used if use_multiprocessing=True)

    Returns:
        List of tuples: (basename, improved_mask_pt_path, overlay_png_path, plot_dir)
    """
    if use_multiprocessing:
        return process_dataset_parallel(
            images_dir,
            masks_dir,
            out_masks_dir,
            out_plots_dir,
            save_as_pt,
            save_visualizations,
            num_workers,
            **improve_kwargs,
        )
    else:
        return process_dataset_sequential(
            images_dir,
            masks_dir,
            out_masks_dir,
            out_plots_dir,
            save_as_pt,
            save_visualizations,
            **improve_kwargs,
        )


# Main execution
if __name__ == "__main__":
    base = "./automated_underwater_area_estimation/data_preprocessed/IBF/"
    images_dir = Path(base) / "images"
    masks_dir = Path(base) / "masks"
    out_base = Path(base)

    out_masks_dir = out_base / "improved_masks"
    out_plots_dir = out_base / "overlays"

    results = process_dataset(
        images_dir,
        masks_dir,
        out_masks_dir,
        out_plots_dir,
        save_as_pt=True,
        save_visualizations=True,
        use_multiprocessing=True,  # Enable parallel processing
        num_workers=4,  # Auto-detect (cpu_count - 1)
        min_obj_frac=0.001,
        min_hole_frac=0.004,
        r_open_frac=0.004,
        r_close_frac=0.004,
        r_recon_frac=0.003,
        keep_largest=False,
    )

    if not results:
        print("No image/mask pairs processed. Please check your directories.")
    else:
        print(f"\nDone. Processed {len(results)} pairs.")
        print(f"Improved masks -> {out_masks_dir}")
        print(f"Overlays per pair -> {out_plots_dir}")
