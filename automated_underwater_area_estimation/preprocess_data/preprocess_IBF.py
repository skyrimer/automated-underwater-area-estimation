import os
import shutil
from pathlib import Path


def parse_cpc_file(cpc_path: str) -> Optional[str]:
    """
    Parse CPC file and extract the referenced image filename.
    
    Reads the first line of a CPC file and extracts the image filename
    from the second comma-separated field.
    
    Args:
        cpc_path: Path to the CPC file to parse
        
    Returns:
        Base filename of the referenced image, or None if parsing fails
    """
    try:
        with open(cpc_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            # Split by comma and find the image path (second field)
            parts = first_line.split(",")
            if len(parts) >= 2:
                # Extract the image path (second field)
                image_path = parts[1].strip('"').rsplit("\\", 1)[1]
                # Get just the filename from the full path
                return os.path.basename(image_path)
    except Exception as e:
        print(f"Error parsing {cpc_path}: {e}")
    return None


def copy_images_and_cpcs(source_folder: str, dest_folder: str) -> None:
    """
    Copy images and CPC files from source to destination,
    organizing them into 'images' and 'cpcs' subfolders.
    Matches are based on the filename referenced in the first line of each CPC file.
    Also copies all images even if they don't have a corresponding CPC file.
    Both CPC and image files use the same naming template based on the image filename.
    """
    os.makedirs("./data_preprocessed", exist_ok=True)
    source_path = Path(source_folder)
    dest_path = Path(dest_folder)

    # Create destination subfolders
    images_dest = dest_path / "images"
    cpcs_dest = dest_path / "cpcs"
    images_dest.mkdir(parents=True, exist_ok=True)
    cpcs_dest.mkdir(parents=True, exist_ok=True)

    copied_pairs_count = 0
    copied_images_only_count = 0
    skipped_count = 0

    # Common image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # Iterate through all subdirectories in IBF folder
    for subfolder in source_path.iterdir():
        if not subfolder.is_dir():
            continue

        print(f"\nProcessing folder: {subfolder.name}")

        # Track which images have been copied with CPC files
        copied_images = set()

        # Find all CPC files in this subfolder
        cpc_files = list(subfolder.glob("*.cpc"))

        # First, process CPC files and their associated images
        for cpc_file in cpc_files:
            if image_filename := parse_cpc_file(cpc_file):
                # Check if the image exists in the source folder
                image_path = subfolder / image_filename

                if image_path.exists():
                    # Create unique names based on image filename template
                    # Both image and CPC use the same base name derived from image filename
                    unique_image_name = f"{subfolder.name}_{image_filename}"

                    # For CPC file, use image filename but with .cpc extension
                    image_name_without_ext = Path(image_filename).stem
                    unique_cpc_name = f"{subfolder.name}_{image_name_without_ext}.cpc"

                    # Copy image
                    shutil.copy2(image_path, images_dest / unique_image_name)
                    # Copy CPC file with new name based on image filename
                    shutil.copy2(cpc_file, cpcs_dest / unique_cpc_name)

                    copied_images.add(image_filename)
                    copied_pairs_count += 1
                    print(f"  ✓ Copied pair: {unique_image_name} and {unique_cpc_name}")
                else:
                    print(
                        f"  ✗ Image not found: {image_filename} (referenced in {cpc_file.name})"
                    )
                    skipped_count += 1
            else:
                print(f"  ✗ Could not parse: {cpc_file.name}")
                skipped_count += 1

        # Now, copy all remaining images that don't have CPC files
        for image_file in subfolder.iterdir():
            if (
                image_file.is_file()
                and image_file.suffix.lower() in image_extensions
                and image_file.name not in copied_images
            ):
                unique_image_name = f"{subfolder.name}_{image_file.name}"
                shutil.copy2(image_file, images_dest / unique_image_name)
                copied_images_only_count += 1
                print(f"  ℹ Copied image only: {unique_image_name}")

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Successfully copied: {copied_pairs_count} pairs (image + CPC)")
    print(f"  Successfully copied: {copied_images_only_count} images without CPC")
    print(f"  Total images copied: {copied_pairs_count + copied_images_only_count}")
    print(f"  Skipped/Failed: {skipped_count} files")
    print(f"  Destination: {dest_folder}")
    print(f"{'='*60}")
