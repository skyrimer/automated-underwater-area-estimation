import json
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from automated_underwater_area_estimation.segmentation.reefsupport.model import (
    ReefSupportModel,
)
from automated_underwater_area_estimation.segmentation.epfl.model import EPFLModel
from automated_underwater_area_estimation.segmentation.segmentation_dataset import (
    CoralSegmentationDataset,
)
from automated_underwater_area_estimation.segmentation.evaluation_metrics import (
    compute_segmentation_metrics,
)


class EvaluationPipeline:
    """Evaluation pipeline with restart support and multi-dataset capabilities."""

    def __init__(
        self, base_results_dir="automated_underwater_area_estimation/evaluation_results"
    ):
        self.base_results_dir = Path(base_results_dir)
        self.base_results_dir.mkdir(parents=True, exist_ok=True)

    def get_dataset_results_dir(self, dataset_name):
        """Get results directory for a specific dataset."""
        return self.base_results_dir / dataset_name

    def get_model_results_dir(self, dataset_name, model_name):
        """Get results directory for a specific model and dataset."""
        return self.get_dataset_results_dir(dataset_name) / model_name

    def load_progress(self, dataset_name, model_name):
        """Load existing progress for a model on a dataset."""
        model_dir = self.get_model_results_dir(dataset_name, model_name)
        progress_file = model_dir / "progress.json"

        if not progress_file.exists():
            return {"completed_indices": [], "last_processed": -1}

        with open(progress_file, "r") as f:
            return json.load(f)

    def save_progress(
        self, dataset_name, model_name, completed_indices, last_processed
    ):
        """Save progress for a model on a dataset."""
        model_dir = self.get_model_results_dir(dataset_name, model_name)
        progress_file = model_dir / "progress.json"

        progress_data = {
            "completed_indices": completed_indices,
            "last_processed": last_processed,
            "timestamp": datetime.now().isoformat(),
        }

        with open(progress_file, "w") as f:
            json.dump(progress_data, f, indent=2)

    def save_inference_result(
        self, dataset_name, model_name, idx, image, gt_mask, pred_mask, metrics
    ):
        """Save individual inference result (mask and metrics)."""
        model_dir = self.get_model_results_dir(dataset_name, model_name)

        # Create subdirectories
        masks_dir = model_dir / "prediction_masks"
        metrics_dir = model_dir / "individual_metrics"
        masks_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Save prediction mask
        # mask_path = masks_dir / f"image_{idx:04d}_prediction.npy"
        # np.save(mask_path, pred_mask.cpu().numpy())

        # Save individual metrics
        result = {
            "image_idx": idx,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "image_shape": (
                list(image.size) if hasattr(image, "size") else list(image.shape)
            ),
        }

        metrics_path = metrics_dir / f"image_{idx:04d}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(result, f, indent=2)

    def load_existing_results(self, dataset_name, model_name):
        """Load existing results from individual metric files."""
        model_dir = self.get_model_results_dir(dataset_name, model_name)
        metrics_dir = model_dir / "individual_metrics"

        if not metrics_dir.exists():
            return []

        results = []
        for metrics_file in sorted(metrics_dir.glob("image_*_metrics.json")):
            with open(metrics_file, "r") as f:
                results.append(json.load(f))

        return results

    def initialize_models(self):
        """Initialize all models for evaluation."""
        models = []

        # ReefSupport models
        reef_models = ["yolov8_sm_latest.pt", "yolov8_xlarge_latest.pt"]
        for model_name in reef_models:
            model = ReefSupportModel(model_name)
            models.append(
                {
                    "name": f"ReefSupport_{model_name.replace('.pt', '')}",
                    "model": model,
                    "type": "ReefSupport",
                }
            )

        # EPFL models
        epfl_models = [
            "EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024",
            "EPFL-ECEO/segformer-b5-finetuned-coralscapes-1024-1024",
        ]
        for model_name in epfl_models:
            model = EPFLModel(model_name)
            models.append(
                {
                    "name": f"EPFL_{model_name.split('/')[-1].split('-')[1]}",
                    "model": model,
                    "type": "EPFL",
                }
            )

        return models

    def create_visualization(self, image, gt_mask, pred_mask, title, save_path=None):
        """Create and optionally save a visualization."""
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Original image
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        axes[0].imshow(img_array)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Ground truth
        gt_array = (
            gt_mask.cpu().numpy() if isinstance(gt_mask, torch.Tensor) else gt_mask
        )
        axes[1].imshow(gt_array, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        # Prediction
        pred_array = (
            pred_mask.cpu().numpy()
            if isinstance(pred_mask, torch.Tensor)
            else pred_mask
        )
        axes[2].imshow(pred_array, cmap="gray")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        # Overlay
        axes[3].imshow(img_array)
        colored_mask = np.zeros((*pred_array.shape, 4))
        colored_mask[pred_array.astype(bool)] = [1, 0, 0, 0.5]  # Red overlay
        axes[3].imshow(colored_mask)
        axes[3].set_title("Overlay")
        axes[3].axis("off")

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            return fig

    def evaluate_single_model(
        self, model_info, dataset, dataset_name, save_images_every=100, resume=True
    ):
        """Evaluate a single model on the dataset with restart support."""

        model_name = model_info["name"]
        model = model_info["model"]

        # Create model-specific directory
        model_dir = self.get_model_results_dir(dataset_name, model_name)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        images_dir = model_dir / "sample_images"
        images_dir.mkdir(exist_ok=True)

        # Load existing progress
        progress = (
            self.load_progress(dataset_name, model_name)
            if resume
            else {"completed_indices": [], "last_processed": -1}
        )
        completed_indices = set(progress["completed_indices"])

        # Load existing results
        existing_results = (
            self.load_existing_results(dataset_name, model_name) if resume else []
        )

        print(f"\nEvaluating {model_name} on {dataset_name}...")
        if resume and completed_indices:
            print(f"Resuming from {len(completed_indices)} completed images...")

        all_metrics = []

        for idx in tqdm(range(len(dataset)), desc=f"Evaluating {model_name}"):
            # Skip if already processed
            if idx in completed_indices:
                continue

            try:
                # Get data
                image, gt_mask = dataset[idx]
                pred_image, pred_mask = model.segment_image(
                    image, adjust_size=False, use_sliding_window=True
                )
                if gt_mask.shape != pred_mask.shape:
                    gt_mask = (
                        F.interpolate(
                            gt_mask.unsqueeze(0)
                            .unsqueeze(0)
                            .float(),  # Add batch and channel dims, convert to float
                            size=pred_mask.shape,  # Match the prediction mask dimensions
                            mode="nearest",
                        )
                        .squeeze()
                        .bool()
                    )
                # Run inference
                # pred_image, pred_mask = model.segment_image(image, adjust_size=True)
                # target_height, target_width = pred_image.size
                # gt_mask = F.interpolate(
                #     gt_mask.unsqueeze(0).unsqueeze(0).float(),  # Add batch and channel dims, convert to float
                #     size=(target_height, target_width),  # Note: interpolate expects (height, width)
                #     mode='nearest'
                # ).squeeze().bool()

                # Compute metrics
                metrics = compute_segmentation_metrics(
                    pred_mask, gt_mask, model.device, metrics="all"
                )

                # Save inference result immediately
                self.save_inference_result(
                    dataset_name, model_name, idx, image, gt_mask, pred_mask, metrics
                )

                # Update progress
                completed_indices.add(idx)
                self.save_progress(
                    dataset_name, model_name, list(completed_indices), idx
                )

                all_metrics.append(metrics)

                # Save sample images
                # if idx % save_images_every == 0 or idx < 10:
                #     viz_path = images_dir / f"image_{idx:04d}_visualization.png"
                #     self.create_visualization(
                #         image,
                #         gt_mask,
                #         pred_mask,
                #         f"{model_name} - {dataset_name} - Image {idx}",
                #         save_path=viz_path,
                #     )

            except Exception as e:
                print(f"Error processing image {idx} for {model_name}: {e}")
                continue

        # Combine existing and new results
        all_results = existing_results + [
            {
                "image_idx": idx,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            }
            for idx, metrics in enumerate(all_metrics)
        ]

        # Sort results by image index
        all_results.sort(key=lambda x: x["image_idx"])

        # Collect all metrics for summary
        all_metrics_for_summary = [result["metrics"] for result in all_results]

        # Compute and save summary statistics
        if all_metrics_for_summary:
            summary_stats = {}
            for metric in all_metrics_for_summary[0].keys():
                values = [m[metric] for m in all_metrics_for_summary]
                summary_stats[f"{metric}_mean"] = float(np.mean(values))
                summary_stats[f"{metric}_std"] = float(np.std(values))
                summary_stats[f"{metric}_min"] = float(np.min(values))
                summary_stats[f"{metric}_max"] = float(np.max(values))

            # Save summary statistics
            summary_path = model_dir / "summary_statistics.json"
            with open(summary_path, "w") as f:
                json.dump(summary_stats, f, indent=2)

            # Create and save summary plots
            self.create_summary_plots(
                all_metrics_for_summary, f"{model_name}_{dataset_name}", model_dir
            )

        # Save detailed results to JSONL file
        jsonl_file = model_dir / "detailed_results.jsonl"
        with open(jsonl_file, "w") as f:
            for result in all_results:
                f.write(json.dumps(result) + "\n")

        # Save model configuration
        config = {
            "model_name": model_name,
            "model_type": model_info["type"],
            "dataset_name": dataset_name,
            "dataset_size": len(dataset),
            "device": str(model.device),
            "evaluation_date": datetime.now().isoformat(),
            "total_processed": len(all_results),
            "total_completed": len(completed_indices),
        }

        config_path = model_dir / "model_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved {len(all_results)} results to {model_dir}")
        print(f"  - Detailed results: {jsonl_file}")
        print(f"  - Summary statistics: {summary_path}")
        print(f"  - Sample images: {images_dir}")
        print(f"  - Prediction masks: {model_dir / 'prediction_masks'}")

        return all_results, summary_stats if all_metrics_for_summary else {}

    def create_summary_plots(self, all_metrics, model_dataset_name, results_dir):
        """Create and save summary plots for a model-dataset combination."""

        plots_dir = results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        metrics_names = list(all_metrics[0].keys())

        # 1. Metrics distribution plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics_names):
            values = [m[metric] for m in all_metrics]
            axes[i].hist(values, bins=30, alpha=0.7, edgecolor="black")
            axes[i].set_title(f"{metric.capitalize()} Distribution")
            axes[i].set_xlabel(f"{metric.capitalize()}")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(True, alpha=0.3)

            # Add statistics text
            mean_val = np.mean(values)
            std_val = np.std(values)
            axes[i].axvline(
                mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.3f}"
            )
            axes[i].legend()

        # Hide the last subplot if odd number of metrics
        if len(metrics_names) < len(axes):
            axes[-1].set_visible(False)

        plt.suptitle(f"{model_dataset_name} - Metrics Distribution")
        plt.tight_layout()
        plt.savefig(
            plots_dir / "metrics_distribution.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        # 2. Metrics over time (image index)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        image_indices = list(range(len(all_metrics)))

        for i, metric in enumerate(metrics_names):
            values = [m[metric] for m in all_metrics]
            axes[i].plot(image_indices, values, alpha=0.7, linewidth=1)
            axes[i].set_title(f"{metric.capitalize()} Over Images")
            axes[i].set_xlabel("Image Index")
            axes[i].set_ylabel(f"{metric.capitalize()}")
            axes[i].grid(True, alpha=0.3)

            # Add trend line
            z = np.polyfit(image_indices, values, 1)
            p = np.poly1d(z)
            axes[i].plot(
                image_indices,
                p(image_indices),
                "r--",
                alpha=0.8,
                label=f"Trend: {z[0]:.6f}x + {z[1]:.3f}",
            )
            axes[i].legend()

        if len(metrics_names) < len(axes):
            axes[-1].set_visible(False)

        plt.suptitle(f"{model_dataset_name} - Metrics Over Images")
        plt.tight_layout()
        plt.savefig(plots_dir / "metrics_over_images.png", dpi=150, bbox_inches="tight")
        plt.close()

    def create_dataset_comparison(self, dataset_name, all_summaries):
        """Create overall comparison across all models for a specific dataset."""

        if not all_summaries:
            print(
                f"No summary data available for comparison on dataset: {dataset_name}"
            )
            return

        # Create comparison directory for this dataset
        comparison_dir = self.get_dataset_results_dir(dataset_name) / "comparison"
        comparison_dir.mkdir(exist_ok=True)

        # Prepare comparison data
        comparison_data = []
        metrics_names = ["dice", "iou", "precision", "recall", "pixel_accuracy"]

        for model_name, summary in all_summaries.items():
            model_row = {"model": model_name, "dataset": dataset_name}
            for metric in metrics_names:
                model_row[f"{metric}_mean"] = summary.get(f"{metric}_mean", 0)
                model_row[f"{metric}_std"] = summary.get(f"{metric}_std", 0)
            comparison_data.append(model_row)

        # Save comparison table
        comparison_file = comparison_dir / "model_comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison_data, f, indent=2)

        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, metric in enumerate(metrics_names):
            model_names = [data["model"] for data in comparison_data]
            means = [data[f"{metric}_mean"] for data in comparison_data]
            stds = [data[f"{metric}_std"] for data in comparison_data]

            x_pos = np.arange(len(model_names))
            bars = axes[i].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
            axes[i].set_xlabel("Models")
            axes[i].set_ylabel(f"{metric.capitalize()}")
            axes[i].set_title(f"{metric.capitalize()} Comparison")
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels(model_names, rotation=45, ha="right")
            axes[i].grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + std + 0.005,
                    f"{mean:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # Hide the last subplot
        axes[-1].set_visible(False)

        plt.suptitle(f"Model Performance Comparison - {dataset_name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            comparison_dir / "models_comparison.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        print(f"Comparison results for {dataset_name} saved to: {comparison_dir}")
        return comparison_data

    def evaluate_on_dataset(self, dataset_path, dataset_name, resume=True):
        """Evaluate all models on a specific dataset."""

        print(f"\n{'=' * 60}")
        print(f"EVALUATING ON DATASET: {dataset_name}")
        print(f"{'=' * 60}")

        # Load dataset
        dataset = CoralSegmentationDataset(dataset_path)
        print(f"Dataset size: {len(dataset)}")

        # Initialize models
        all_models = self.initialize_models()
        print(f"Initialized {len(all_models)} models:")
        for model_info in all_models:
            print(f"  - {model_info['name']} ({model_info['type']})")

        # Evaluate each model
        all_results = {}
        all_summaries = {}

        for model_info in all_models:
            try:
                results, summary = self.evaluate_single_model(
                    model_info, dataset, dataset_name, resume=resume
                )
                all_results[model_info["name"]] = results
                all_summaries[model_info["name"]] = summary
            except Exception as e:
                print(f"Failed to evaluate {model_info['name']} on {dataset_name}: {e}")
                continue

        # Create dataset comparison
        comparison_data = self.create_dataset_comparison(dataset_name, all_summaries)

        # Print summary for this dataset
        print(f"\n{'=' * 60}")
        print(f"EVALUATION SUMMARY - {dataset_name}")
        print(f"{'=' * 60}")

        for model_name, summary in all_summaries.items():
            print(f"\n{model_name}:")
            print("-" * 50)

            metrics_names = ["dice", "iou", "precision", "recall", "pixel_accuracy"]
            for metric in metrics_names:
                mean_val = summary.get(f"{metric}_mean", 0)
                std_val = summary.get(f"{metric}_std", 0)
                print(f"  {metric:15}: {mean_val:.4f} ± {std_val:.4f}")

        return all_results, all_summaries, comparison_data


# Usage example - evaluate on multiple datasets
def main():
    # Initialize evaluation pipeline
    pipeline = EvaluationPipeline()

    # Define datasets to evaluate on
    datasets = {
        "coralscop_test": "./automated_underwater_area_estimation/data_preprocessed/coralscop/test",
        # Add more datasets here as needed
        # "coralscop_val": "./automated_underwater_area_estimation/data_preprocessed/coralscop/val",
        # "another_dataset": "/path/to/another/dataset",
    }

    # Evaluate on each dataset
    all_dataset_results = {}

    for dataset_name, dataset_path in datasets.items():
        if os.path.exists(dataset_path):
            results, summaries, comparison = pipeline.evaluate_on_dataset(
                dataset_path, dataset_name, resume=True
            )
            all_dataset_results[dataset_name] = {
                "results": results,
                "summaries": summaries,
                "comparison": comparison,
            }
        else:
            print(f"Dataset path does not exist: {dataset_path}")

    # Create cross-dataset summary
    print(f"\n{'=' * 80}")
    print("CROSS-DATASET SUMMARY")
    print(f"{'=' * 80}")

    for dataset_name, data in all_dataset_results.items():
        print(f"\nDataset: {dataset_name}")
        print("-" * 60)

        for model_name, summary in data["summaries"].items():
            metrics_names = ["dice", "iou", "precision", "recall", "pixel_accuracy"]
            metrics_str = " | ".join(
                [
                    f"{metric}: {summary.get(f'{metric}_mean', 0):.3f}"
                    for metric in metrics_names[:3]
                ]
            )
            print(f"  {model_name:30}: {metrics_str}")


if __name__ == "__main__":
    main()
