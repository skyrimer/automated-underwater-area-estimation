import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_and_process_experiment_folders(base_path):
    """
    Find folders containing both summary_statistics.json and model_config.json files.
    Extract the required information from both files.
    """
    results = []
    base_path = Path(base_path)

    # Find all summary_statistics.json files
    summary_files = list(base_path.rglob("summary_statistics.json"))

    for summary_file in summary_files:
        folder_path = summary_file.parent
        model_config_file = folder_path / "model_config.json"

        # Check if both files exist in the same folder
        if model_config_file.exists():
            try:
                # Read model config
                with open(model_config_file, "r") as f:
                    model_config = json.load(f)

                # Read summary statistics
                with open(summary_file, "r") as f:
                    summary_stats = json.load(f)

                # Extract required information
                result = {
                    "folder_path": str(folder_path),
                    "model_name": model_config.get("model_name", "N/A"),
                    "dataset_name": model_config.get("dataset_name", "N/A"),
                    "device": model_config.get("device", "N/A"),
                    "dataset_size": model_config.get("dataset_size", "N/A"),
                    **summary_stats,  # Add all metrics from summary statistics
                }

                results.append(result)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing files in {folder_path}: {e}")

    return results


# Process the experiment folders
base_directory = (
    r".\automated_underwater_area_estimation\evaluation_results_vm\evaluation_results"
)
experiment_data = find_and_process_experiment_folders(base_directory)

# Create a DataFrame for better visualization and analysis
df = pd.DataFrame(experiment_data)

# Get metric columns (excluding config columns) - focus on mean metrics
config_columns = ["folder_path", "model_name", "dataset_name", "device", "dataset_size"]
all_metric_columns = [col for col in df.columns if col not in config_columns]
mean_metric_columns = [col for col in all_metric_columns if col.endswith("_mean")]

# Plot metrics for each model with mean and standard deviation
print("\n" + "=" * 80)
print("GENERATING METRIC PLOTS")
print("=" * 80)

# Get metric columns (excluding config columns)
config_columns = ["folder_path", "model_name", "dataset_name", "device", "dataset_size"]
metric_columns = [
    col
    for col in df.columns
    if col not in config_columns and df[col].dtype in ["float64", "int64"]
]

# Focus on main performance metrics (mean values only, exclude std/min/max)
main_metrics = [col for col in metric_columns if col.endswith("_mean")]
std_metrics = [col.replace("_mean", "_std") for col in main_metrics]
available_std_metrics = [col for col in std_metrics if col in df.columns]


# Calculate statistics by model using existing mean and std columns
model_stats = {}
if len(df) > 0 and main_metrics:
    for metric in main_metrics:
        # Get corresponding std column name
        std_metric = metric.replace("_mean", "_std")

        if std_metric in df.columns:
            # Use the recorded mean and std values from CSV
            model_data = []
            for model in df["model_name"].unique():
                model_rows = df[df["model_name"] == model]
                # Calculate average of the mean and std values across all experiments for this model
                mean_of_means = model_rows[metric].mean()
                # Average the std values
                avg_std = model_rows[std_metric].mean()
                model_data.append(
                    {"model_name": model, "mean": mean_of_means, "std": avg_std}
                )
            model_stats[metric] = pd.DataFrame(model_data)
        else:
            # Fallback to calculating if std column doesn't exist
            stats = df.groupby("model_name")[metric].agg(["mean", "std"]).reset_index()
            stats["std"] = stats["std"].fillna(0)
            model_stats[metric] = stats

    # Update metric_columns to use only main metrics for plotting
    metric_columns = main_metrics

    # Create plots - determine subplot layout
    n_metrics = len(metric_columns)
    # Calculate subplot grid
    cols = min(3, n_metrics)  # Max 3 columns
    rows = (n_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = [axes] if rows == 1 and cols == 1 else axes.flatten()
    for i, metric in enumerate(metric_columns):
        ax = axes[i] if i < len(axes) else axes[-1]

        stats = model_stats[metric]

        # Create bar plot
        x_pos = np.arange(len(stats))
        bars = ax.bar(
            x_pos,
            stats["mean"],
            yerr=stats["std"],
            capsize=5,
            alpha=0.7,
            color="skyblue",
            edgecolor="navy",
            linewidth=1,
        )

        # Customize the plot
        ax.set_xlabel("Model")
        ax.set_ylabel(f'{metric.replace("_", " ").title()}')
        ax.set_title(f'{metric.replace("_", " ").title()} by Model')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stats["model_name"], rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, mean_val, std_val in zip(bars, stats["mean"], stats["std"]):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{mean_val:.3f}±{std_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    # Hide empty subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        "./report_visualisations/metrics_by_model.png", dpi=300, bbox_inches="tight"
    )

    print("Plots saved as: ./report_visualisations/metrics_by_model.png")
else:
    print("No data available for plotting or no numeric metrics found.")

# Create overview plots for each dataset
print("\n" + "=" * 80)
print("GENERATING DATASET OVERVIEW PLOTS")
print("=" * 80)

if len(df) > 0 and metric_columns:
    # Get unique datasets
    datasets = df["dataset_name"].unique()
    print(f"Creating overview plots for {len(datasets)} datasets: {list(datasets)}")

    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")

        # Filter data for this dataset
        dataset_df = df[df["dataset_name"] == dataset]

        # Prepare data using existing mean and std columns from CSV
        dataset_stats = {}
        for metric in metric_columns:
            # Get corresponding std column name
            std_metric = metric.replace("_mean", "_std")

            if std_metric in dataset_df.columns:
                # Use the recorded mean and std values from CSV
                model_data = []
                for model in dataset_df["model_name"].unique():
                    model_rows = dataset_df[dataset_df["model_name"] == model]
                    # For each model, use the mean and std values directly from the CSV
                    mean_val = model_rows[metric].iloc[0] if len(model_rows) > 0 else 0
                    std_val = (
                        model_rows[std_metric].iloc[0] if len(model_rows) > 0 else 0
                    )
                    model_data.append(
                        {"model_name": model, "mean": mean_val, "std": std_val}
                    )
                dataset_stats[metric] = pd.DataFrame(model_data)
            else:
                # Fallback to calculating if std column doesn't exist
                stats = (
                    dataset_df.groupby("model_name")[metric]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                stats["std"] = stats["std"].fillna(0)
                dataset_stats[metric] = stats

        # Create subplot layout for all metrics in this dataset
        n_metrics = len(metric_columns)
        cols = min(3, n_metrics)  # Max 3 columns
        rows = (n_metrics + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        fig.suptitle(
            f"Performance Overview - {dataset} Dataset",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        axes = [axes] if rows == 1 and cols == 1 else axes.flatten()
        # Color scheme for different models
        colors = ["skyblue", "lightcoral", "lightgreen", "orange", "purple"]

        for i, metric in enumerate(metric_columns):
            ax = axes[i] if i < len(axes) else axes[-1]

            stats = dataset_stats[metric]

            # Create bar plot with error bars
            x_pos = np.arange(len(stats))
            bars = ax.bar(
                x_pos,
                stats["mean"],
                yerr=stats["std"],
                capsize=8,
                alpha=0.8,
                color=[colors[j % len(colors)] for j in range(len(stats))],
                edgecolor="black",
                linewidth=1,
                error_kw={"linewidth": 2, "capthick": 2},
            )

            # Customize the plot
            ax.set_xlabel("Model", fontsize=10, fontweight="bold")
            ax.set_ylabel(
                f'{metric.replace("_", " ").title()}', fontsize=10, fontweight="bold"
            )
            ax.set_title(
                f'{metric.replace("_", " ").title()}', fontsize=11, fontweight="bold"
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels(stats["model_name"], rotation=45, ha="right", fontsize=9)
            ax.grid(True, alpha=0.3, linestyle="--")

            # Add value labels on bars with mean±std format
            for bar, mean_val, std_val in zip(bars, stats["mean"], stats["std"]):
                height = bar.get_height()
                # Position label above the error bar
                label_y = (
                    height + std_val + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                )
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    label_y,
                    f"{mean_val:.3f}±{std_val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

            # Highlight best performing model
            best_idx = stats["mean"].idxmax()
            bars[best_idx].set_edgecolor("gold")
            bars[best_idx].set_linewidth(3)

        # Hide empty subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        # Add dataset summary text
        summary_text = f"Dataset: {dataset}\n"
        summary_text += f"Models: {len(dataset_df['model_name'].unique())}\n"
        summary_text += f"Experiments: {len(dataset_df)}\n"
        summary_text += f"Avg Dataset Size: {dataset_df['dataset_size'].mean():.0f}"

        fig.text(
            0.02,
            0.02,
            summary_text,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.15)

        # Save dataset overview plot
        plot_filename = f"./report_visualisations/{dataset}_overview.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")

        print(f"Dataset overview plot saved as: {plot_filename}")
