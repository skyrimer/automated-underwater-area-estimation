import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def load_detailed_results(base_path):
    results = []
    base_path = Path(base_path)
    detailed_files = list(base_path.rglob("detailed_results.jsonl"))
    for file_path in detailed_files:
        print(f"Processing: {file_path}")
        print(file_path)
        if "ReefSupport" in file_path.__str__():
            continue
        try:
            with open(file_path, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        flattened_data = {
                            "image_idx": data["image_idx"],
                            "model_name": data["model_name"],
                            "dataset_name": data["dataset_name"],
                            "timestamp": data["timestamp"],
                            **data["metrics"],
                        }
                        results.append(flattened_data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing file {file_path}: {e}")
    return results


# ==== Load data ====
base_directory = r".\automated_underwater_area_estimation\evaluation_results"
detailed_results = load_detailed_results(base_directory)
df = pd.DataFrame(detailed_results)
# figure out metric columns dynamically
non_metric_columns = ["image_idx", "model_name", "dataset_name", "timestamp"]
metric_columns = [c for c in df.columns if c not in non_metric_columns]

# ensure output dir exists
out_dir = Path("")
out_dir.mkdir(parents=True, exist_ok=True)

# ==== Compute per-dataset, per-model stats ====
dataset_stats = []
for dataset in sorted(df["dataset_name"].unique()):
    ds_data = df[df["dataset_name"] == dataset]
    if len(ds_data) == 0:
        continue
    for model in sorted(ds_data["model_name"].unique()):
        model_data = ds_data[ds_data["model_name"] == model]
        stats_row = {
            "dataset_name": dataset,
            "model_name": model,
            "total_images": len(model_data),
        }
        for metric in metric_columns:
            if metric in model_data.columns:
                stats_row[f"{metric}_mean"] = model_data[metric].mean()
                stats_row[f"{metric}_std"] = model_data[metric].std()
                stats_row[f"{metric}_min"] = model_data[metric].min()
                stats_row[f"{metric}_max"] = model_data[metric].max()
        dataset_stats.append(stats_row)

dataset_df = pd.DataFrame(dataset_stats)

# Save dataset statistics to CSV
dataset_summary_csv = out_dir / "dataset_statistics_summary.csv"
dataset_df.to_csv(dataset_summary_csv, index=False)
print(f"\nDataset statistics saved to: {dataset_summary_csv}")

# Also save the full detailed results (unaltered)
detailed_output_file = out_dir / "detailed_results.csv"
df.to_csv(detailed_output_file, index=False)
print(f"Detailed results saved to: {detailed_output_file}")

# ==== Create per-dataset comparison plots (grouped by model) ====
print("\n" + "=" * 80)
print("GENERATING DATASET-LEVEL COMPARISON PLOTS")
print("=" * 80)

width = 0.25  # bar width; adjust if you have many models
for metric in metric_columns:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in dataset_df.columns:
        continue

    plt.figure(figsize=(20, 6))

    # consistent ordering on x-axis
    unique_datasets = sorted(dataset_df["dataset_name"].unique())
    unique_models = sorted(dataset_df["model_name"].unique())
    x = np.arange(len(unique_datasets))

    # draw bars per model
    for i, model in enumerate(unique_models):
        model_means = []
        model_stds = []
        model_slice = dataset_df[dataset_df["model_name"] == model]

        # align values to x order
        for ds in unique_datasets:
            row = model_slice[model_slice["dataset_name"] == ds]
            if not row.empty and pd.notna(row.iloc[0].get(mean_col, np.nan)):
                model_means.append(row.iloc[0][mean_col])
                model_stds.append(row.iloc[0].get(std_col, 0.0))
            else:
                model_means.append(0.0)
                model_stds.append(0.0)

        bars = plt.bar(
            x + i * width,
            model_means,
            width,
            yerr=model_stds,
            label=model,
            alpha=0.85,
            capsize=4,
            error_kw={"linewidth": 1.5, "capthick": 1.5},
        )

        # labels above bars (only for nonzero means)
        local_max = max(model_means) if len(model_means) else 0.0
        bump = (local_max * 0.02) if local_max > 0 else 0.0
        # for bar, mean_val, std_val in zip(bars, model_means, model_stds):
        #     if mean_val > 0:
        #         height = bar.get_height()
        #         plt.text(
        #             bar.get_x() + bar.get_width() / 2.0,
        #             height + std_val + bump,
        #             f"{mean_val:.3f}±{std_val:.3f}",
        #             ha="center",
        #             va="bottom",
        #             fontsize=8,
        #             fontweight="bold",
        #         )

    plt.xlabel("Dataset", fontsize=12, fontweight="bold")
    plt.ylabel(metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
    plt.title(
        f"{metric.replace('_', ' ').title()} Performance by Dataset",
        fontsize=14,
        fontweight="bold",
    )
    # center ticks across grouped bars
    plt.xticks(
        x + (len(unique_models) - 1) * width / 2.0,
        unique_datasets,
        rotation=30,
        ha="right",
    )
    plt.legend(ncol=min(len(unique_models), 4))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_filename = out_dir / f"dataset_{metric}_comparison.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved as: {plot_filename}")
