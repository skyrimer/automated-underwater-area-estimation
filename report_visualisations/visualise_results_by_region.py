import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_detailed_results(base_path):
    """
    Load all detailed_results.jsonl files from the evaluation results directory.
    """
    results = []
    base_path = Path(base_path)

    # Find all detailed_results.jsonl files
    detailed_files = list(base_path.rglob("detailed_results.jsonl"))

    for file_path in detailed_files:
        print(f"Processing: {file_path}")
        if "ReefSupport" in file_path.__str__() or "EPFL_b2" in file_path.__str__():
            continue
        try:
            with open(file_path, "r") as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        data = json.loads(line)
                        # Flatten the metrics into the main dictionary
                        flattened_data = {
                            "image_idx": data["image_idx"],
                            "model_name": data["model_name"],
                            "dataset_name": data["dataset_name"],
                            "timestamp": data["timestamp"],
                            **data["metrics"],  # Flatten metrics
                        }
                        results.append(flattened_data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing file {file_path}: {e}")

    return results


# Define regional groupings
REGIONAL_GROUPS = {
    "Caribbean": [
        "SEAFLOWER_BOLIVAR",
        "SEAFLOWER_COURTOWN",
        "TETES_PROVIDENCIA",
        "UNAL_BLEACHING_TAYRONA",
    ],
    "Atlantic (non-Caribbean)": ["SEAVIEW_ATL"],
    "Indo-Pacific": ["SEAVIEW_IDN_PHL"],
    "Pacific – Australia": ["SEAVIEW_PAC_AUS"],
    "Pacific – USA": ["SEAVIEW_PAC_USA"],
}


def add_regional_grouping(df):
    """
    Add regional grouping column to the dataframe based on dataset names.
    """

    def get_region(dataset_name):
        for region, datasets in REGIONAL_GROUPS.items():
            if dataset_name in datasets:
                return region
        return "unknown"

    df["region"] = df["dataset_name"].apply(get_region)
    return df


# Load all detailed results
base_directory = r".\automated_underwater_area_estimation\evaluation_results"
detailed_results = load_detailed_results(base_directory)


# Add regional groupings
df = add_regional_grouping(pd.DataFrame(detailed_results))

# Get metric columns (exclude non-metric columns)
non_metric_columns = ["image_idx", "model_name", "dataset_name", "timestamp", "region"]
metric_columns = [col for col in df.columns if col not in non_metric_columns]

# Calculate statistics by region and model
regional_stats = []

for region in REGIONAL_GROUPS.keys():
    region_data = df[df["region"] == region]

    if len(region_data) > 0:
        for model in region_data["model_name"].unique():
            model_data = region_data[region_data["model_name"] == model]

            stats_row = {
                "region": region,
                "model_name": model,
                "total_images": len(model_data),
            }

            # Calculate mean and std for each metric
            for metric in metric_columns:
                if metric in model_data.columns:
                    stats_row[f"{metric}_mean"] = model_data[metric].mean()
                    stats_row[f"{metric}_std"] = model_data[metric].std()
                    stats_row[f"{metric}_min"] = model_data[metric].min()
                    stats_row[f"{metric}_max"] = model_data[metric].max()

            regional_stats.append(stats_row)
# Create regional statistics DataFrame
regional_df = pd.DataFrame(regional_stats)

# Create visualization for regional comparison
print("\n" + "=" * 80)
print("GENERATING REGIONAL COMPARISON PLOTS")
print("=" * 80)

# Create plots for each metric comparing regions
width = 0.25
for metric in metric_columns:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    plt.figure(figsize=(15, 6))

    # Prepare data for plotting
    regions = []
    models = []
    means = []
    stds = []

    for _, row in regional_df.iterrows():
        if mean_col in row and pd.notna(row[mean_col]):
            regions.append(row["region"].replace("_", " ").title())
            models.append(row["model_name"])
            means.append(row[mean_col])
            stds.append(row[std_col])

    # Create grouped bar plot
    unique_regions = list(set(regions))
    unique_models = list(set(models))

    x = np.arange(len(unique_regions))

    colors = ["skyblue", "lightcoral", "lightgreen"]

    for i, model in enumerate(unique_models):
        model_means = []
        model_stds = []

        for region in unique_regions:
            if region_model_data := [
                (m, s)
                for r, mo, m, s in zip(regions, models, means, stds)
                if r == region and mo == model
            ]:
                model_means.append(region_model_data[0][0])
                model_stds.append(region_model_data[0][1])
            else:
                model_means.append(0)
                model_stds.append(0)

        bars = plt.bar(
            x + i * width,
            model_means,
            width,
            yerr=model_stds,
            label=model,
            alpha=0.8,
            capsize=5,
            color=colors[i % len(colors)],
            error_kw={"linewidth": 2, "capthick": 2},
        )

        # Add value labels on bars
        for bar, mean_val, std_val in zip(bars, model_means, model_stds):
            if mean_val > 0:  # Only add labels for non-zero values
                height = bar.get_height()
                # Position label above the error bar
                label_y = height + std_val + (max(model_means) * 0.02)
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    label_y,
                    f"{mean_val:.3f}±{std_val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    rotation=0,
                )

    plt.xlabel("Region", fontsize=12, fontweight="bold")
    plt.ylabel(f"{metric.title()}", fontsize=12, fontweight="bold")
    plt.title(f"{metric.title()} Performance by Region", fontsize=14, fontweight="bold")
    plt.xticks(x + width, unique_regions)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_filename = f"./report_visualisations/regional_{metric}_comparison.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")

    print(f"Plot saved as: {plot_filename}")

# Save regional statistics to CSV
output_file = "./report_visualisations/regional_statistics_summary.csv"
regional_df.to_csv(output_file, index=False)
print(f"\nRegional statistics saved to: {output_file}")

# Also save the full detailed results with regional groupings
detailed_output_file = "./report_visualisations/detailed_results_with_regions.csv"
df.to_csv(detailed_output_file, index=False)
print(f"Detailed results with regional groupings saved to: {detailed_output_file}")
