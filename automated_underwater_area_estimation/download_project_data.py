from automated_underwater_area_estimation.preprocess_data.preprocess_IBF import (
    copy_images_and_cpcs,
)
from automated_underwater_area_estimation.preprocess_data.preprocess_reefsupport import (
    process_all_reef_support_folders,
)

from google.cloud import storage
import os
from tqdm.auto import tqdm


def download_gcs_folder(
    bucket_name: str, source_folder: str, destination_folder: str
) -> None:
    """
    Download all files from a Google Cloud Storage (GCS) folder to local directory.

    Connects to a public GCS bucket anonymously and downloads all files
    from the specified folder prefix to the local destination.

    Args:
        bucket_name: Name of the GCS bucket
        source_folder: Folder prefix in the bucket to download from
        destination_folder: Local directory path to save downloaded files
    """

    # Initialize client (no authentication needed for public buckets)
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)

    # List all blobs in the folder
    blobs = bucket.list_blobs(prefix=source_folder)

    os.makedirs(destination_folder, exist_ok=True)

    for blob in tqdm(blobs):
        # Skip if it's just a folder marker
        if blob.name.endswith("/"):
            continue

        # Create local file path
        local_file_path = os.path.join(
            destination_folder, os.path.relpath(blob.name, source_folder)
        )

        # Create directory if it doesn't exist
        local_dir = os.path.dirname(local_file_path)
        os.makedirs(local_dir, exist_ok=True)

        # Download the file
        blob.download_to_filename(local_file_path)

    print("Download completed!")


def main() -> None:
    """
    Main function to download and preprocess project data.

    Downloads IBF and reef_support datasets from GCS bucket,
    then preprocesses them into the required format.
    """
    package_name = "automated_underwater_area_estimation"
    bucket = "rs_storage_open"
    gcs_folders = [
        ("IBF", "point_labels"),
        ("reef_support", "mask_labels"),
    ]
    for gcs_folder in gcs_folders:
        dataset, dataset_type = gcs_folder
        download_gcs_folder(
            bucket,
            source_folder=f"benthic_datasets/{dataset_type}/{dataset}/",
            destination_folder=f"./{package_name}/data/{dataset}",
        )

    copy_images_and_cpcs(
        source_folder=f"./{package_name}/data/IBF",
        dest_folder=f"./{package_name}/data_preprocessed/IBF",
    )

    process_all_reef_support_folders(
        source_path=f"./{package_name}/data/reef_support",
        output_base_path=f"./{package_name}/data_preprocessed/reef_support",
    )
    # Obtain masks for the IBF dataset
    # morphologically improve the masks
    # augment the masks
    # train the model
    # done


if __name__ == "__main__":
    main()
