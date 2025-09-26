import os
import json
import shutil
import argparse
from typing import List, Tuple, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
from pycocotools import mask as coco_mask


def load_coco_annotation(annotation_data: dict) -> Tuple[List[torch.Tensor], dict]:
    """
    Load COCO format annotation and extract masks

    Args:
        annotation_data: Dictionary containing COCO annotation data

    Returns:
        Tuple of (list of mask tensors, image info dictionary)
    """
    image_info = annotation_data["image"]
    annotations = annotation_data["annotations"]
    masks = []

    for i, ann in enumerate(annotations):
        # Decode RLE mask
        rle = ann["segmentation"]
        mask = coco_mask.decode(rle)

        # Convert to tensor
        mask_tensor = torch.from_numpy(mask).long()
        masks.append(mask_tensor)

    return masks, image_info


def combine_masks_union(mask_list: List[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Combine multiple binary masks into a single mask using union (OR operation)

    Args:
        mask_list: List of tensors with 1s and 0s, all same shape

    Returns:
        Single tensor with 1 if any mask has 1 at that position, 0 otherwise (uint8)
    """
    if not mask_list:
        return None

    # Stack masks along a new dimension
    stacked_masks = torch.stack(mask_list, dim=0)  # Shape: (num_masks, H, W)

    # Use uint8 for binary data (8x smaller than default long)
    return torch.any(stacked_masks.bool(), dim=0).to(torch.uint8)


def save_image_mask_pairs(
    source_json_path: str,
    source_image_path: str,
    output_path: str,
    copy_images: bool = True,
    verbose: bool = True,
) -> int:
    """
    Save image-mask pairs in an optimal format for later loading

    Args:
        source_json_path: Path to JSON files
        source_image_path: Path to corresponding images
        output_path: Path to save processed data
        copy_images: Whether to copy images or create symlinks
        verbose: Whether to print detailed progress information

    Returns:
        Number of successfully processed image-mask pairs
    """
    # Create output directories
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "masks"), exist_ok=True)

    processed_count = 0
    skipped_count = 0
    error_count = 0

    json_files = [f for f in os.listdir(source_json_path) if f.endswith(".json")]

    if verbose:
        print(f"Processing {len(json_files)} JSON files...")

    progress_bar = tqdm(json_files) if verbose else json_files

    for coco_json in progress_bar:
        coco_json_path = os.path.join(source_json_path, coco_json)

        # Check if file is empty
        if os.path.getsize(coco_json_path) == 0:
            if verbose:
                print(f"Skipping empty file: {coco_json}")
            skipped_count += 1
            continue

        try:
            # Load annotation data
            with open(coco_json_path, "r") as f:
                annotation_data = json.load(f)

            # Get image filename from annotation
            image_filename = annotation_data["image"]["file_name"]
            image_path = os.path.join(source_image_path, image_filename)

            # Check if corresponding image exists
            if not os.path.exists(image_path):
                if verbose:
                    print(f"Image not found: {image_filename}")
                skipped_count += 1
                continue

            # Process masks
            masks, _ = load_coco_annotation(annotation_data)
            if not masks:
                if verbose:
                    print(f"No masks found in: {coco_json}")
                skipped_count += 1
                continue

            union_mask = combine_masks_union(masks)
            if union_mask is None:
                skipped_count += 1
                continue

            # Get base filename without extension
            base_name = os.path.splitext(image_filename)[0]

            # Handle image file
            output_image_path = os.path.join(output_path, "images", image_filename)
            if not os.path.exists(output_image_path):
                if copy_images:
                    shutil.copy2(image_path, output_image_path)
                else:
                    # Create symlink (saves space)
                    os.symlink(os.path.abspath(image_path), output_image_path)

            # Save mask as numpy array
            mask_filename = f"{base_name}.npy"
            mask_path = os.path.join(output_path, "masks", mask_filename)
            np.save(mask_path, union_mask.numpy())

            processed_count += 1

        except json.JSONDecodeError as e:
            if verbose:
                print(f"JSON decode error in {coco_json}: {e}")
            error_count += 1
            continue
        except Exception as e:
            if verbose:
                print(f"Error processing {coco_json}: {e}")
            error_count += 1
            continue

    if verbose:
        print(f"\nProcessing complete:")
        print(f"  Successfully processed: {processed_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"  Errors: {error_count}")

    return processed_count


def save_metadata(output_path: str, processed_count: int, source_paths: dict) -> None:
    """
    Save processing metadata

    Args:
        output_path: Path where data was saved
        processed_count: Number of processed samples
        source_paths: Dictionary of source paths used
    """
    metadata = {
        "total_samples": processed_count,
        "source_json_path": source_paths["json"],
        "source_image_path": source_paths["image"],
        "output_path": output_path,
        "processing_info": {
            "mask_format": "numpy uint8 arrays",
            "mask_values": "Binary (0/1)",
            "image_format": "Original format preserved",
        },
    }

    metadata_path = os.path.join(output_path, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {metadata_path}")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Process COCO format coral segmentation data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--source-json-path",
        type=str,
        required=True,
        help="Path to directory containing JSON annotation files",
    )

    parser.add_argument(
        "--source-image-path",
        type=str,
        required=True,
        help="Path to directory containing corresponding image files",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save processed image-mask pairs",
    )

    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of creating symlinks (uses more disk space)",
    )

    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="Save processing metadata to output directory",
    )

    args = parser.parse_args()

    # Validate input paths
    if not os.path.exists(args.source_json_path):
        print(f"Error: JSON source path does not exist: {args.source_json_path}")
        return 1

    if not os.path.exists(args.source_image_path):
        print(f"Error: Image source path does not exist: {args.source_image_path}")
        return 1

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    verbose = not args.quiet

    if verbose:
        print("Coral Mask Processor")
        print("===================")
        print(f"JSON source: {args.source_json_path}")
        print(f"Image source: {args.source_image_path}")
        print(f"Output: {args.output_path}")
        print(f"Copy images: {args.copy_images}")
        print()

    # Process the data
    processed_count = save_image_mask_pairs(
        source_json_path=args.source_json_path,
        source_image_path=args.source_image_path,
        output_path=args.output_path,
        copy_images=args.copy_images,
        verbose=verbose,
    )

    # Save metadata if requested
    if args.save_metadata:
        source_paths = {"json": args.source_json_path, "image": args.source_image_path}
        save_metadata(args.output_path, processed_count, source_paths)

    if processed_count == 0:
        print("Warning: No files were successfully processed!")
        return 1

    return 0


if __name__ == "__main__":
    main()
