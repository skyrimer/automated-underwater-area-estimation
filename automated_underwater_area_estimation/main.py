from automated_underwater_area_estimation.download.download_gcs_bucket import (
    download_gcs_folder,
)
from automated_underwater_area_estimation.preprocess_data.preprocess_segmentation_validation import (
    save_image_mask_pairs,
)
from automated_underwater_area_estimation.preprocess_data.preprocess_data import (
    copy_images_and_cpcs,
)
from automated_underwater_area_estimation.preprocess_data.preprocess_reefsupport import (
    process_all_reef_support_folders
)

package_name = "automated_underwater_area_estimation"
bucket = "rs_storage_open"
gcs_folders = [
    # ("IBF", "point_labels"),
    # ("reef_support", "mask_labels"),
    ("Coralseg", "mask_labels"),
    ("coralscop_masks", "mask_labels"),
]
# for gcs_folder in gcs_folders:
#     download_gcs_folder(
#         bucket,
#         source_folder=f"benthic_datasets/{gcs_folder[1]}/{gcs_folder[0]}/",
#         destination_folder=f"./{package_name}/data/{gcs_folder[0]}",
#     )

# for split in ["train", "test"]:
#     source_json_path = f"./{package_name}/data/coralscop_masks/{split}/jsons/"
#     source_image_path = f"./{package_name}/data/coralscop_masks/{split}/images/"
#     output_path = f"./{package_name}/data_preprocessed/coralscop/{split}"

#     save_image_mask_pairs(source_json_path, source_image_path, output_path)


# source_folder = f"./{package_name}/data/IBF"
# dest_folder = f"./{package_name}/data_preprocessed/IBF"
# copy_images_and_cpcs(source_folder, dest_folder)


process_all_reef_support_folders(
    source_path=f"./{package_name}/data/reef_support",
    output_base_path=f"./{package_name}/data_preprocessed/reef_support",
)