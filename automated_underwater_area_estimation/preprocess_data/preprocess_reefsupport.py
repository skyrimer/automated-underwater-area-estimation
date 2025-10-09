import os
import json
import shutil
import argparse
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from tqdm.auto import tqdm
from PIL import Image
import cv2


def load_visual_mask(mask_image_path: str, color_threshold: int = 0) -> Optional[torch.Tensor]:
    """
    Load visual mask image and convert to boolean mask
    
    Args:
        mask_image_path: Path to mask image file
        color_threshold: Minimum color intensity to consider as mask (0-255)
        
    Returns:
        Boolean tensor with True for masked regions (any non-zero pixel), None if error
    """
    try:
        # Load image in RGB format
        mask_img = Image.open(mask_image_path).convert('RGB')
        mask_array = np.array(mask_img)
        
        # Extract RGB channels
        red_channel = mask_array[:, :, 0]
        green_channel = mask_array[:, :, 1] 
        blue_channel = mask_array[:, :, 2]
        
        # Create mask where any channel is above threshold (non-zero by default)
        # True if any channel > threshold, False only if all channels are 0
        combined_mask = (red_channel > color_threshold) | \
                       (green_channel > color_threshold) | \
                       (blue_channel > color_threshold)
        
        # Convert to tensor
        return torch.from_numpy(combined_mask).to(torch.uint8)
        
    except Exception as e:
        print(f"Error loading visual mask {mask_image_path}: {e}")
        return None


def find_corresponding_mask(image_filename: str, masks_path: str) -> Optional[str]:
    """
    Find corresponding mask file for an image
    
    Args:
        image_filename: Name of the image file
        masks_path: Path to masks directory
        
    Returns:
        Mask filename if found, None otherwise
    """
    # Get base name without extension
    base_name = os.path.splitext(image_filename)[0]
    
    # Look for mask with "_mask" suffix and various extensions
    mask_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    
    for ext in mask_extensions:
        mask_filename = f"{base_name}_mask{ext}"
        mask_path = os.path.join(masks_path, mask_filename)
        if os.path.exists(mask_path):
            return mask_filename
    
    return None


def process_reef_support_folder(
    folder_path: str,
    output_path: str,
    copy_images: bool = True,
    color_threshold: int = 50,
    verbose: bool = True
) -> int:
    """
    Process a single reef support folder containing images and masks_stitched
    
    Args:
        folder_path: Path to folder containing 'images' and 'masks_stitched' subfolders
        output_path: Path to save processed data
        copy_images: Whether to copy images or create symlinks
        color_threshold: Color threshold for mask detection
        verbose: Whether to print progress information
        
    Returns:
        Number of successfully processed image-mask pairs
    """
    images_path = os.path.join(folder_path, "images")
    masks_path = os.path.join(folder_path, "masks_stitched")
    
    # Validate input folders exist
    if not os.path.exists(images_path):
        if verbose:
            print(f"Images folder not found: {images_path}")
        return 0
        
    if not os.path.exists(masks_path):
        if verbose:
            print(f"Masks folder not found: {masks_path}")
        return 0
    
    # Create output directories
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "masks"), exist_ok=True)
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # Get list of image files
    image_files = [f for f in os.listdir(images_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    
    if verbose:
        print(f"Processing {len(image_files)} images from {os.path.basename(folder_path)}...")
    
    progress_bar = tqdm(image_files, desc=f"Processing {os.path.basename(folder_path)}") if verbose else image_files
    
    for image_filename in progress_bar:
        try:
            # Get base name without extension
            base_name = os.path.splitext(image_filename)[0]
            
            # Look for corresponding mask file with "_mask" suffix
            mask_filename = find_corresponding_mask(image_filename, masks_path)
            
            if mask_filename is None:
                if verbose:
                    print(f"No corresponding mask found for: {image_filename}")
                skipped_count += 1
                continue
            
            # Load and process mask
            mask_path = os.path.join(masks_path, mask_filename)
            mask_tensor = load_visual_mask(mask_path, color_threshold)
            
            if mask_tensor is None:
                error_count += 1
                continue
            
            # Handle image file
            source_image_path = os.path.join(images_path, image_filename)
            output_image_path = os.path.join(output_path, "images", image_filename)
            
            if not os.path.exists(output_image_path):
                if copy_images:
                    shutil.copy2(source_image_path, output_image_path)
                else:
                    # Create symlink (saves space)
                    os.symlink(os.path.abspath(source_image_path), output_image_path)
            
            # Save mask as numpy array
            mask_output_filename = f"{base_name}.npy"
            mask_output_path = os.path.join(output_path, "masks", mask_output_filename)
            np.save(mask_output_path, mask_tensor.numpy())
            
            processed_count += 1
            
        except Exception as e:
            if verbose:
                print(f"Error processing {image_filename}: {e}")
            error_count += 1
            continue
    
    if verbose:
        print(f"Folder {os.path.basename(folder_path)} complete:")
        print(f"  Successfully processed: {processed_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"  Errors: {error_count}")
    
    return processed_count


def process_all_reef_support_folders(
    source_path: str,
    output_base_path: str,
    copy_images: bool = True,
    color_threshold: int = 50,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Process all folders in the reef support directory
    
    Args:
        source_path: Path to 'data/reef_support' directory
        output_base_path: Base path for output (e.g., 'data_preprocessed')
        copy_images: Whether to copy images or create symlinks
        color_threshold: Color threshold for mask detection
        verbose: Whether to print progress information
        
    Returns:
        Dictionary mapping folder names to number of processed samples
    """
    if not os.path.exists(source_path):
        raise ValueError(f"Source path does not exist: {source_path}")
    
    # Get all subdirectories
    folder_names = [f for f in os.listdir(source_path) 
                   if os.path.isdir(os.path.join(source_path, f))]
    
    if not folder_names:
        print(f"No folders found in: {source_path}")
        return {}
    
    results = {}
    total_processed = 0
    
    if verbose:
        print(f"Found {len(folder_names)} folders to process:")
        print(f"  {', '.join(folder_names)}")
        print()
    
    for folder_name in folder_names:
        folder_path = os.path.join(source_path, folder_name)
        output_path = os.path.join(output_base_path, folder_name)
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Process this folder
        processed_count = process_reef_support_folder(
            folder_path=folder_path,
            output_path=output_path,
            copy_images=copy_images,
            color_threshold=color_threshold,
            verbose=verbose
        )
        
        results[folder_name] = processed_count
        total_processed += processed_count
        
        if verbose:
            print()
    
    if verbose:
        print("="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        for folder_name, count in results.items():
            print(f"{folder_name}: {count} samples")
        print(f"\nTotal processed samples: {total_processed}")
    
    return results


def save_processing_metadata(
    output_base_path: str, 
    results: Dict[str, int], 
    source_path: str,
    processing_params: dict
) -> None:
    """
    Save processing metadata for all folders
    
    Args:
        output_base_path: Base output path
        results: Results dictionary from processing
        source_path: Original source path
        processing_params: Parameters used for processing
    """
    metadata = {
        "total_folders_processed": len(results),
        "total_samples": sum(results.values()),
        "folder_results": results,
        "source_path": source_path,
        "output_base_path": output_base_path,
        "processing_params": processing_params,
        "processing_info": {
            "mask_format": "numpy uint8 arrays",
            "mask_values": "Binary (0/1)",
            "mask_source": "Red/blue visual masks converted to boolean",
            "mask_naming": "Masks have '_mask' suffix before extension",
            "image_format": "Original format preserved",
        }
    }
    
    metadata_path = os.path.join(output_base_path, "processing_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Processing metadata saved to: {metadata_path}")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Process reef support data with visual masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--source-path",
        type=str,
        default="data/reef_support",
        help="Path to reef support directory containing folders with images and masks_stitched"
    )
    
    parser.add_argument(
        "--output-path", 
        type=str,
        default="data_preprocessed",
        help="Base path for output preprocessed data"
    )
    
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of creating symlinks (uses more disk space)"
    )
    
    parser.add_argument(
        "--color-threshold",
        type=int,
        default=50,
        help="Color intensity threshold for mask detection (0-255)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true", 
        help="Suppress verbose output"
    )
    
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="Save processing metadata to output directory"
    )
    
    args = parser.parse_args()
    
    # Validate input path
    if not os.path.exists(args.source_path):
        print(f"Error: Source path does not exist: {args.source_path}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    verbose = not args.quiet
    
    if verbose:
        print("Reef Support Data Processor")
        print("===========================")
        print(f"Source: {args.source_path}")
        print(f"Output: {args.output_path}")
        print(f"Copy images: {args.copy_images}")
        print(f"Color threshold: {args.color_threshold}")
        print(f"Mask naming: [image_name]_mask.[ext]")
        print()
    
    # Process all folders
    results = process_all_reef_support_folders(
        source_path=args.source_path,
        output_base_path=args.output_path,
        copy_images=args.copy_images,
        color_threshold=args.color_threshold,
        verbose=verbose
    )
    
    # Save metadata if requested
    if args.save_metadata:
        processing_params = {
            "copy_images": args.copy_images,
            "color_threshold": args.color_threshold
        }
        save_processing_metadata(args.output_path, results, args.source_path, processing_params)
    
    total_processed = sum(results.values())
    if total_processed == 0:
        print("Warning: No files were successfully processed!")
        return 1
    
    return 0


if __name__ == "__main__":
    main()