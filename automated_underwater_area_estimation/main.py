from automated_underwater_area_estimation.segmentation_corals.epfl.model import (
    EPFLModel,
)
from automated_underwater_area_estimation.segmentation_quadrant.model import (
    QuadrantSegmentationModel,
)
from automated_underwater_area_estimation.area_estimation.area_estimation import (
    estimate_area_using_quadrant,
)
from pathlib import Path
from PIL import Image
import time
import numpy as np
import argparse


def process_image(
    image_path_str: str,
    quadrant_width: float = 54,
    quadrant_height: float = 54,
    verbose: bool = False,
):
    """
    Processes the given image through the pipeline, measures time for each step,
    and returns the coral segmentation mask, quadrant segmentation mask, and PAE.

    Times for each step are printed.

    Args:
        image_path_str (str): Path to the image file.
        quadrant_width (float): Width of the quadrant in cm (default: 54).
        quadrant_height (float): Height of the quadrant in cm (default: 54).
        verbose (bool): Verbosity for time measurements.

    Returns:
        tuple: (coral_segmentation_mask, quadrant_segmentation_mask, pae)
    """
    # Load image
    start_time = time.time()
    image_path = Path(image_path_str)
    if not image_path.exists():
        raise FileNotFoundError(f"Image {image_path} does not exist")
    image = Image.open(image_path)
    image_load_time = time.time() - start_time
    if verbose:
        print(f"Image loading time: {image_load_time:.2f} seconds")

    # Load models
    start_time = time.time()
    coral_segmentation_model = EPFLModel(
        "EPFL-ECEO/segformer-b5-finetuned-coralscapes-1024-1024"
    )
    quadrant_segmentation_model = QuadrantSegmentationModel()
    model_load_time = time.time() - start_time
    if verbose:
        print(f"Model loading time: {model_load_time:.2f} seconds")

    # Coral segmentation
    start_time = time.time()
    _, coral_segmentation_mask = coral_segmentation_model.segment_image(
        image, adjust_size=False, use_sliding_window=True
    )
    coral_seg_time = time.time() - start_time
    if verbose:
        print(f"Coral segmentation time: {coral_seg_time:.2f} seconds")

    # Quadrant segmentation
    start_time = time.time()
    quadrant_segmentation_mask = quadrant_segmentation_model.segment_image(image)
    quadrant_seg_time = time.time() - start_time
    if verbose:
        print(f"Quadrant segmentation time: {quadrant_seg_time:.2f} seconds")

    # Area estimation
    start_time = time.time()
    pae = estimate_area_using_quadrant(
        quadrant_segmentation_mask, quadrant_width, quadrant_height
    )
    area_est_time = time.time() - start_time
    if verbose:
        print(f"Area estimation time: {area_est_time:.2f} seconds")

    return coral_segmentation_mask, quadrant_segmentation_mask, pae


def create_weighted_mask_counts(
    coral_segmentation_mask, pae: float, verbose: bool = False
):
    """
    Takes a binary coral segmentation mask and a PAE value, counts True and False pixels,
    multiplies each count by PAE, and stores results in a dictionary.

    Args:
        coral_segmentation_mask: Binary mask (numpy array with values 0 or 1, or True/False).
        pae: The projected area estimate value (scalar).
        verbose (bool): Verbosity for printing results.

    Returns:
        dict: Dictionary containing counts of True and False pixels, their weighted values,
              and the modified mask.
    """
    # Ensure mask is a numpy array
    mask = np.array(coral_segmentation_mask.cpu(), dtype=bool)

    # Count True and False pixels
    true_count = np.sum(mask)
    false_count = np.sum(~mask)

    # Store results in dictionary
    result_dict = {
        "coral_pixel_count": int(true_count),
        "background_pixel_count": int(false_count),
        "coral_area_cm2": true_count * pae,
        "background_area_cm2": false_count * pae,
    }

    if verbose:
        print(f"Coral pixel count: {result_dict['coral_pixel_count']}")
        print(f"Background pixel count: {result_dict['background_pixel_count']}")
        print(f"Coral area (cm²): {result_dict['coral_area_cm2']:.2f}")
        print(f"Background area (cm²): {result_dict['background_area_cm2']:.2f}")

    return result_dict


def main():
    parser = argparse.ArgumentParser(
        description="Process an underwater image for coral segmentation and area estimation."
    )
    parser.add_argument("image_path", type=str, help="Path to the input image file")
    parser.add_argument(
        "--quadrant-width",
        type=float,
        default=54,
        help="Width of the quadrant in cm (default: 54)",
    )
    parser.add_argument(
        "--quadrant-height",
        type=float,
        default=54,
        help="Height of the quadrant in cm (default: 54)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for timing and results",
    )

    args = parser.parse_args()
    verbose = args.verbose
    image_path = args.image_path

    # Process the image
    print(f"Processing image: {image_path}")
    coral_mask, quadrant_mask, pae = process_image(
        image_path,
        quadrant_width=args.quadrant_width,
        quadrant_height=args.quadrant_height,
        verbose=verbose,
    )

    # Create weighted mask counts
    results = create_weighted_mask_counts(coral_mask, pae, verbose=verbose)

    # Print final results if not verbose (verbose mode already prints in create_weighted_mask_counts)
    if not verbose:
        print(f"Results for {image_path}:")
        print(f"Coral pixel count: {results['coral_pixel_count']}")
        print(f"Background pixel count: {results['background_pixel_count']}")
        print(f"Coral area (cm²): {results['coral_area_cm2']:.2f}")
        print(f"Background area (cm²): {results['background_area_cm2']:.2f}")


if __name__ == "__main__":
    main()
