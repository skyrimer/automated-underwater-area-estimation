import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict

from automated_underwater_area_estimation.segmentation_quadrant.model import (
    QuadrantSegmentationModel,
)
from automated_underwater_area_estimation.area_estimation.quadrant_setup.area_estimation import (
    median_band_average_from_cm_ratio,
)

class Evaluator:
    def __init__(
        self,
        csv_path: Path,
        image_root: Path = None,
    ):
        """
        model_checkpoint: path to trained segmentation model
        csv_path: path to CSV file containing ground-truth columns including at least:
            image_path, pixel_area_gt_cm^2
        image_root: optional base directory for image paths in csv
        """
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.model = QuadrantSegmentationModel()

    def evaluate_and_save(
        self,
        output_csv_path: Path,
        save_predictions: bool = True
    ) -> Dict[str, float]:
        """
        Runs evaluation over all rows, computes summary metrics,
        and if save_predictions is True, saves a CSV with predictions and ground truth for each image.

        Returns summary metrics dictionary.
        """
        results: List[Tuple[str, float, float, float]] = []
        for idx, row in self.df.iterrows():
            img_path = row["image_path"]
            if self.image_root is not None:
                img_path = (self.image_root / img_path).resolve()
            image = Image.open(img_path).convert("RGB")

            # segmentation & estimation
            mask = self.model.segment_image(image)
            pred_area = median_band_average_from_cm_ratio(mask, 54, 54, 0.08)
            gt_area = float(row["pixel_area_gt_cm^2"])
            abs_err = abs(pred_area - gt_area)
            rel_err = abs_err / gt_area if gt_area != 0 else float("nan")

            results.append((str(img_path), pred_area, gt_area, abs_err, rel_err))

        # build DataFrame of per‐image results
        cols = ["image_path", "pred_area_cm2", "gt_area_cm2", "abs_error_cm2", "rel_error_fraction"]
        results_df = pd.DataFrame(results, columns=cols)

        if save_predictions:
            results_df.to_csv(output_csv_path, index=False, encoding="utf-8")
            print(f"Saved predictions to {output_csv_path}")

        # summary metrics
        preds = results_df["pred_area_cm2"].to_numpy(dtype=float)
        gts   = results_df["gt_area_cm2"].to_numpy(dtype=float)
        abs_errors = results_df["abs_error_cm2"].to_numpy(dtype=float)
        rel_errors = results_df["rel_error_fraction"].to_numpy(dtype=float)

        mae           = float(np.mean(abs_errors))
        rmse          = float(np.sqrt(np.mean((preds - gts)**2)))
        bias          = float(np.mean(preds - gts))
        r2            = 1.0 - (np.sum((preds - gts)**2) / np.sum((gts - np.mean(gts))**2)) if gts.size > 1 else float("nan")
        mean_rel_err  = float(np.nanmean(rel_errors))
        median_rel_err= float(np.nanmedian(rel_errors))

        summary = {
            "N_samples"           : int(len(gts)),
            "MAE_cm2"             : mae,
            "RMSE_cm2"            : rmse,
            "Bias_cm2"            : bias,
            "R2"                  : r2,
            "MeanRelError_fraction": mean_rel_err,
            "MedianRelError_fraction": median_rel_err,
        }

        # optional: print per‐image results
        for _, row in results_df.iterrows():
            print(f"{row['image_path']} | pred: {row['pred_area_cm2']:.4f} cm² | "
                  f"gt: {row['gt_area_cm2']:.4f} cm² | abs_err: {row['abs_error_cm2']:.4f} | "
                  f"rel_err: {row['rel_error_fraction']*100:.2f}%")

        print("SUMMARY:", summary)
        return summary

if __name__ == "__main__":
    csv_path = Path(__file__).parent / "quadrant_points.csv"
    evaluator = Evaluator(
        csv_path=csv_path,
        image_root=None,
    )
    out_csv = Path(__file__).parent / "quadrant_predictions.csv"
    summary = evaluator.evaluate_and_save(output_csv_path=out_csv, save_predictions=True)
