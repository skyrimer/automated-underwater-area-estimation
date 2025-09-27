from automated_underwater_area_estimation.download.download_gcs_bucket import (
    download_gcs_folder,
)
from automated_underwater_area_estimation.preprocess_data.preprocess_segmentation_validation import (
    save_image_mask_pairs,
)

package_name = "automated_underwater_area_estimation"
bucket = "rs_storage_open"
gcs_folders = [
    ("coralscop_masks", "mask_labels"),
    ("IBF", "point_labels"),
]
for gcs_folder in gcs_folders:
    download_gcs_folder(
        bucket,
        source_folder=f"benthic_datasets/{gcs_folder[1]}/{gcs_folder[0]}/",
        destination_folder=f"./{package_name}/data/{gcs_folder[0]}",
    )

split = "test"
source_json_path = f"./{package_name}/data/coralscop_masks/{split}/jsons/"
source_image_path = f"./{package_name}/data/coralscop_masks/{split}/images/"
output_path = f"./{package_name}/data_preprocessed/coralscop/{split}"

save_image_mask_pairs(source_json_path, source_image_path, output_path)
