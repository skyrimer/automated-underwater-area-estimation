from automated_underwater_area_estimation.download_gcs_bucket import (
    download_gcs_folder,
)

from automated_underwater_area_estimation.preprocess_data.preprocess_IBF import (
    copy_images_and_cpcs,
)
from automated_underwater_area_estimation.preprocess_data.preprocess_reefsupport import (
    process_all_reef_support_folders,
)
from automated_underwater_area_estimation.segmentation_corals.segmentation_evaluation import (
    main as segmentation_evaluation,
)


def main():
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
    segmentation_evaluation()
    # Obtain masks for the IBF dataset
    # morphologically improve the masks
    # augment the masks
    # train the model
    # done


if __name__ == "__main__":
    main()
