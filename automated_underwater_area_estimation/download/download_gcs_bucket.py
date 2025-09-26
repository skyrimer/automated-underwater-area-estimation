from google.cloud import storage
import os
import argparse
from tqdm.auto import tqdm


def download_gcs_folder(bucket_name, source_folder, destination_folder):
    """Download all files from a GCS folder to local directory."""

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


def main():
    parser = argparse.ArgumentParser(
        description="Download files from a GCS bucket folder to local directory"
    )

    parser.add_argument(
        "--bucket_name", help="Name of the GCS bucket"
    )

    parser.add_argument(
        "--source_folder",
        help="Source folder path in the GCS bucket",
    )

    parser.add_argument(
        "--destination",
        help="Local destination folder path",
    )

    args = parser.parse_args()

    download_gcs_folder(args.bucket_name, args.source_folder, args.destination)


if __name__ == "__main__":
    main()
