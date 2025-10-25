# Automated Underwater Area Estimation

A comprehensive machine learning pipeline for automated coral reef area estimation from underwater images. This system combines state-of-the-art deep learning models to segment coral reefs and detect reference quadrants, enabling precise measurement of coral coverage in cm².

---

## Table of Contents

- [Project Overview](#project-overview)
  - [What Does This System Do?](#what-does-this-system-do)
  - [Key Features](#key-features)
  - [Architecture](#architecture)
- [Prerequisites](#prerequisites)
  - [System Requirements](#system-requirements)
  - [Check Your Python Version](#check-your-python-version)
- [Installation](#installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Install Poetry](#2-install-poetry)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. GPU Support (Optional but Recommended)](#4-gpu-support-optional-but-recommended)
- [Complete Pipeline Guide](#complete-pipeline-guide)
  - [Phase 1: Data Acquisition](#phase-1-data-acquisition)
    - [Step 1.1: Download Dataset from Google Cloud Storage](#step-11-download-dataset-from-google-cloud-storage)
  - [Phase 2: Coral Segmentation Evaluation](#phase-2-coral-segmentation-evaluation)
    - [Step 2.1: Evaluate Coral Segmentation Models](#step-21-evaluate-coral-segmentation-models)
  - [Phase 3: Quadrant Segmentation Training](#phase-3-quadrant-segmentation-training)
    - [Step 3.1: Label Images for Quadrant Segmentation](#step-31-label-images-for-quadrant-segmentation)
    - [Step 3.2: Preprocess and Improve Raw Masks](#step-32-preprocess-and-improve-raw-masks)
    - [Step 3.3: Augment Training Data](#step-33-augment-training-data)
    - [Step 3.4: Train Quadrant Segmentation Model](#step-34-train-quadrant-segmentation-model)
  - [Phase 4: Area Estimation Evaluation](#phase-4-area-estimation-evaluation)
    - [Step 4.1: Generate Ground Truth Area Values](#step-41-generate-ground-truth-area-values)
    - [Step 4.2: Evaluate Area Estimation Accuracy](#step-42-evaluate-area-estimation-accuracy)
  - [Phase 5: Results Visualization](#phase-5-results-visualization)
    - [Step 5.1: Generate Evaluation Plots (All Datasets)](#step-51-generate-evaluation-plots-all-datasets)
    - [Step 5.2: Generate Regional Comparison Plots](#step-52-generate-regional-comparison-plots)
- [Quick Start: Processing a Single Image](#quick-start-processing-a-single-image)
- [Development](#development)
  - [Code Formatting](#code-formatting)
  - [Running Tests](#running-tests)
  - [Adding Dependencies](#adding-dependencies)
  - [Interactive Development](#interactive-development)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
  - [Code Style](#code-style)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

### What Does This System Do?

This project automates the measurement of coral coverage in underwater photographs using a two-stage deep learning approach:

1. **Coral Segmentation**: Identifies coral vs. non-coral pixels using existing segmentation models (EPFL-ECEO, ReefSupport, CoralScop)
2. **Quadrant Detection**: Locates the physical reference frame (sampling quadrant) in the image using a custom-trained Segformer model
3. **Area Estimation**: Converts pixel measurements to real-world area (cm²) using the detected quadrant as a reference scale

### Key Features

- **Multiple Segmentation Models**: Supports EPFL Segformer (b2/b5), ReefSupport YOLO, and CoralSCOP models
- **Sliding Window Inference**: Handles high-resolution images efficiently with overlapping window processing
- **Robust Quadrant Detection**: Custom training pipeline with morphological mask refinement and data augmentation
- **Comprehensive Evaluation**: Multi-dataset evaluation with boundary-aware metrics (Boundary IoU, Boundary F1)
- **Interactive Labeling**: Streamlit-based annotation tool using AquaSAM for quadrant mask generation
- **Automatic Device Detection**: Seamlessly runs on CUDA, MPS (Apple Silicon), XPU (Intel), or CPU

### Architecture

The system operates in two independent stages:

```
Underwat image
      │
      ├──────────────────────┬──────────────────────┐
      │                      │                      │
      ▼                      ▼                      ▼
  Coral Segmentation   Quadrant Detection     (Original)
  (EPFLModel/YOLO)    (QuadrantSegModel)          │
      │                      │                      │
      ▼                      ▼                      │
  Binary Mask          Binary Mask                 │
  (coral/non-coral)    (quadrant/background)       │
      │                      │                      │
      └──────────────────────┴──────────────────────┘
                             │
                             ▼
                     Area Estimation
                     (Corner Detection + PAE)
                             │
                             ▼
                    Coral Coverage (cm²)
```

---

## Prerequisites

### System Requirements

- **Python**: >=3.12, <3.13
- **Operating Systems**: Windows, macOS, Linux
- **Hardware**: 
  - CPU: Any modern processor (multi-core recommended)
  - GPU: NVIDIA GPU with CUDA support (recommended), Apple Silicon, or Intel GPU
  - RAM: 16GB minimum, 32GB recommended for large datasets
  - Disk: 50GB+ free space for datasets and model weights

### Check Your Python Version

```bash
python --version  # or python3 --version
```

You should see `Python 3.12.x`.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/automated-underwater-area-estimation.git
cd automated-underwater-area-estimation
```

### 2. Install Poetry

If you don't have Poetry installed:

```bash
# On Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# On macOS/Linux
curl -sSL https://install.python-poetry.org | python3 -
```

See the [official Poetry installation guide](https://python-poetry.org/docs/#installation) for more options.

### 3. Install Dependencies

```bash
# Install all dependencies (CPU version of PyTorch)
poetry install

# Activate the virtual environment
poetry shell
```

### 4. GPU Support (Optional but Recommended)

#### For NVIDIA GPUs with CUDA suppport on Windows:

```bash
# Add PyTorch CUDA source (adjust CUDA version as needed)
poetry source add --priority=explicit pytorch-cu129 https://download.pytorch.org/whl/cu129
poetry add --group cuda --source pytorch-cu129 torch torchvision
poetry install --with cuda
```

#### All other systems:
If there is a GPU available, the pipeline will switch to it. Otherwise, next best device will be used.

---

## Complete Pipeline Guide

This section walks through the entire workflow from data download to final visualizations.

### Phase 1: Data Acquisition

#### Step 1.1: Download Dataset from Google Cloud Storage

The project uses two datasets: IBF and ReefSupport, both stored in a public GCS bucket.

```bash
poetry run python automated_underwater_area_estimation/download_project_data.py
```

**What this does:**
- Downloads IBF point labels and ReefSupport mask labels from `gs://rs_storage_open/benthic_datasets/`
- Saves to `automated_underwater_area_estimation/data/IBF/` and `data/reef_support/`
- Automatically preprocesses both datasets:
  - **IBF**: Copies images and CPC files to `data_preprocessed/IBF/`
  - **ReefSupport**: Converts COCO-format annotations to separate image/mask pairs in `data_preprocessed/reef_support/`

**Directory structure after download:**
```
automated_underwater_area_estimation/
├── data/
│   ├── IBF/              # Raw IBF data
│   └── reef_support/     # Raw ReefSupport data
└── data_preprocessed/
    ├── IBF/
    │   ├── images/       # Underwater photos (.JPG)
    │   └── cpcs/         # Coral Point Count files (.cpc)
    └── reef_support/
        ├── images/
        └── masks/
```

---

### Phase 2: Coral Segmentation Evaluation

#### Step 2.1: Evaluate Coral Segmentation Models

Evaluate pre-trained coral segmentation models (EPFL Segformer, ReefSupport YOLO, CoralSCOP) across multiple datasets.

```bash
poetry run python automated_underwater_area_estimation/segmentation_corals/segmentation_evaluation.py
```

**What this does:**
- Loads 4 models:
  - `EPFL_b2`: Segformer-b2 fine-tuned on CoralScapes (1024×1024)
  - `EPFL_b5`: Segformer-b5 fine-tuned on CoralScapes (1024×1024)
  - `ReefSupport_yolov8_sm_latest`: YOLOv8 small
  - `ReefSupport_yolov8_xlarge_latest`: YOLOv8 xlarge
  - `CoralSCOP`: Alternative segmentation model
- Evaluates on multiple datasets that belong to ReefSupport mask dataset (CoralScapes, SeaFlower, Seaview, etc.)
- Computes metrics: IoU, Dice, Boundary IoU, Boundary F1
- Saves per-image predictions and metrics to `evaluation_results/<dataset>/<model>/`
- Supports resumable execution (tracks progress in `progress.json`)

**Output structure:**
```
automated_underwater_area_estimation/
└── evaluation_results/
    ├── CoralScapes/
    │   ├── EPFL_b2/
    │   │   ├── prediction_masks/     # .npy files
    │   │   ├── individual_metrics/    # per-image .json
    │   │   ├── detailed_results.jsonl
    │   │   ├── summary.json
    │   │   └── progress.json
    │   ├── EPFL_b5/
    │   └── ...
    └── SEAFLOWER_BOLIVAR/
        └── ...
```

**Expected runtime:** 2-10 hours depending on device and the system.

---

### Phase 3: Quadrant Segmentation Training

Train a custom Segformer model to detect the sampling quadrant in underwater images.

#### Step 3.1: Label Images for Quadrant Segmentation

Use the interactive Streamlit app with AquaSAM (Segment Anything Model fine-tuned on underwater data) to create quadrant masks.

**Prerequisites:**
- Download AquaSAM weights: Place `aquasam_weights.pth` in `automated_underwater_area_estimation/label_studio/`
- Ensure images are in `data_preprocessed/IBF/images/`

```bash
poetry run streamlit run automated_underwater_area_estimation/label_studio/quadrant_mask_labelling.py
```

**How to use the labeling tool:**

1. **Select Image**: Navigate through unlabeled images
2. **Click Points**: 
   - Green clicks = include this region (positive)
   - Red clicks = exclude this region (negative)
3. **Preview Mask**: SAM generates segmentation in real-time
4. **Save**: Saves mask as `.pt` file in `data_preprocessed/IBF/masks/`
5. **Next**: Move to next image

**Output:**
- Binary masks: `data_preprocessed/IBF/masks/*.pt` (torch tensors)
- Click coordinates: `data_preprocessed/IBF/clicks/*.json` (for reproducibility)

**Tips:**
- Start with 4 corner points on the quadrant
- Add positive clicks inside quadrant if under-segmented
- Add negative clicks outside quadrant if over-segmented
- Use "Clear Clicks" to restart if needed

---

#### Step 3.2: Preprocess and Improve Raw Masks

Apply morphological operations to clean up SAM-generated masks.

```bash
poetry run python automated_underwater_area_estimation/segmentation_quadrant/preprocess_raw_masks.py
```

**What this does:**
- Loads raw masks from `data_preprocessed/IBF/masks/`
- Applies morphological cleaning:
  - Keeps largest connected component
  - Removes small objects (speckles)
  - Fills small holes
  - Binary opening + opening-by-reconstruction
  - Binary closing + closing-by-reconstruction
- Saves improved masks to `data_preprocessed/IBF/improved_masks/`
- Generates side-by-side comparison images in `data_preprocessed/IBF/overlays/`
- Uses multiprocessing for speed (configurable workers)

**Configuration (edit script if needed):**
```python
min_obj_frac = 0.001   # Remove objects <0.1% of image area
min_hole_frac = 0.004  # Fill holes <0.4% of image area
r_open_frac = 0.004    # Opening radius = 0.4% of min(height, width)
r_close_frac = 0.004   # Closing radius = 0.4% of min(height, width)
keep_largest = False   # Keep all components or only largest?
```

**Output:**
- Improved masks: `data_preprocessed/IBF/improved_masks/*_improved.pt`
- Visualizations: `data_preprocessed/IBF/overlays/*_comparison.png` (300 DPI)

---

#### Step 3.3: Augment Training Data

Generate augmented training samples from labeled images to improve model robustness.

```bash
poetry run python automated_underwater_area_estimation/segmentation_quadrant/data_augmentation.py
```

**Augmentation strategy:**
- **Fixed rotations**: 0°, +15°, -15° (3 angles)
- **Per rotation**: 5 randomized augmentations
- **Total multiplier**: 15× dataset size

**Augmentations applied (per variant):**
- Small random affine transforms (translate ±5%, scale ±10%, rotate ±3°)
- Horizontal flip (50% chance)
- Brightness/contrast adjustment (±25%)
- Hue/saturation shift (underwater lighting variability)
- Blur (Gaussian, Motion, or Median)
- Gaussian noise (simulates turbidity)
- Coarse dropout (1-6 holes, 3-10% size, simulates occlusions)
- Resize to 800×600 (training resolution)

**Configuration (edit script if needed):**
```python
ANGLES = [0, 15, -15]           # Fixed rotation angles
AUGS_PER_ANGLE = 5              # Random variants per angle
OUTPUT_SIZE = (800, 600)        # Training resolution (W×H)
```

**Output:**
- Augmented images: `data_preprocessed/IBF/out_images/*_rot{angle}_aug{n}.jpg`
- Augmented masks: `data_preprocessed/IBF/out_masks/*_rot{angle}_aug{n}.pt`

**Example filenames:**
```
IMG_0001_rot000_aug00.jpg  # Original, variant 0
IMG_0001_rot000_aug01.jpg  # Original, variant 1
IMG_0001_rot015_aug00.jpg  # +15° rotation, variant 0
IMG_0001_rot-15_aug00.jpg  # -15° rotation, variant 0
```

---

#### Step 3.4: Train Quadrant Segmentation Model

Train a Segformer model on the augmented dataset.

```bash
poetry run python automated_underwater_area_estimation/segmentation_quadrant/train_model.py
```

**Training configuration:**
- **Architecture**: Segformer (MIT-B0 encoder from NVIDIA)
- **Input size**: 800×600 (resized during data loading)
- **Classes**: 2 (background=0, quadrant=1)
- **Loss function**: Weighted Cross-Entropy + Soft Dice Loss
  - CE weights computed via Median Frequency Balancing
  - Dice weight: 1.0
- **Optimizer**: AdamW (lr=6e-5, weight decay=0.01)
- **Scheduler**: Cosine annealing
- **Epochs**: 100 (early stopping via best validation mIoU)
- **Batch size**: 24 (adjust based on GPU memory)
- **Precision**: bfloat16 (if CUDA available)

**Data split strategy:**
- **Validation ratio**: 20%
- **Group-aware splitting**: All augmentations of the same original image stay in the same split (no leakage)
- **Dataset-stratified**: Ensures balanced representation from IBF and ReefSupport datasets

**Evaluation metrics:**
- Mean IoU (mIoU)
- Dice coefficient
- Boundary IoU (edge-aware, 1-pixel band)
- Boundary F1 (tolerance=2 pixels)

**Output:**
- Training checkpoints: `segmentation_quadrant/checkpoints/segformer_quad_preaug/`
- Best model: `segmentation_quadrant/segformer_best/` (used by inference pipeline)
- Split files: `data_preprocessed/IBF/splits/train.txt`, `val.txt`

**Expected runtime:** 6-20 hours on GPU, and too many hours to be estimated on CPU.

**Monitoring training:**
```bash
# Watch training logs
tail -f segmentation_quadrant/checkpoints/segformer_quad_preaug/trainer_state.json
```

---

### Phase 4: Area Estimation Evaluation

#### Step 4.1: Generate Ground Truth Area Values

Extract quadrant corner coordinates from CPC files and compute ground truth pixel-to-cm² conversion factors.

```bash
poetry run python automated_underwater_area_estimation/area_estimation/ground_truth_generation.py
```

**What this does:**
- Parses `.cpc` files (Coral Point Count format) from `data_preprocessed/IBF/cpcs/`
- Extracts ROI corners (C1, C2, C3, C4) in "working" coordinate space
- Scales corners to pixel coordinates based on image dimensions
- Computes quadrant dimensions:
  - Width: average of top and bottom edges (in pixels)
  - Height: average of left and right edges (in pixels)
- Calculates ground truth pixel area:
  - `pixel_area_gt_cm² = (52.5 / width_px) * (52.5 / height_px)`
  - Assumes quadrant is 52.5 cm × 52.5 cm (configurable)
- Saves results to CSV with columns:
  - `stem`, `image_path`, `cpc_path`
  - `c1x`, `c1y`, ..., `c4x`, `c4y` (corner coordinates)
  - `quadrant_width_px`, `quadrant_height_px`
  - `pixel_width_cm`, `pixel_height_cm`
  - `pixel_area_gt_cm^2` (ground truth PAE)

**Output:**
- CSV file: `area_estimation/quadrant_points.csv`

---

#### Step 4.2: Evaluate Area Estimation Accuracy

Run the full pipeline (quadrant detection + area estimation) and compare to ground truth.

```bash
poetry run python automated_underwater_area_estimation/area_estimation/evaluation.py
```

**What this does:**
- Loads images and ground truth from `quadrant_points.csv`
- For each image:
  1. Runs quadrant segmentation model
  2. Detects quadrant corners (TL, TR, BR, BL) using geometric algorithms
  3. Computes predicted PAE via median filtering of 6 edge/diagonal distances
- Compares predicted vs. ground truth PAE
- Computes evaluation metrics:
  - **MAE** (Mean Absolute Error in cm²)
  - **RMSE** (Root Mean Squared Error)
  - **Bias** (systematic over/under-estimation)
  - **R²** (coefficient of determination)
  - **Mean/Median Relative Error** (fraction)
- Saves per-image predictions to CSV

**Output:**
- Predictions CSV: `area_estimation/quadrant_predictions.csv`
  - Columns: `image_path`, `pred_area_cm2`, `gt_area_cm2`, `abs_error_cm2`, `rel_error_fraction`
- Console output:
  ```
  SUMMARY: {'N_samples': 150, 'MAE_cm2': 0.0023, 'RMSE_cm2': 0.0031, 
            'Bias_cm2': -0.0001, 'R2': 0.987, 
            'MeanRelError_fraction': 0.034, 'MedianRelError_fraction': 0.027}
  ```

---

### Phase 5: Results Visualization

#### Step 5.1: Generate Evaluation Plots (All Datasets)

Create comparison plots for coral segmentation performance across all evaluated datasets.

```bash
poetry run python report_visualisations/visualise_results.py
```

**What this does:**
- Loads all `detailed_results.jsonl` files from `evaluation_results/*/`
- Aggregates metrics by dataset and model
- Generates grouped bar charts for each metric:
  - X-axis: Datasets
  - Y-axis: Metric value (mean ± std)
  - Bars: Grouped by model
- Saves plots to project root

**Output:**
- CSV files:
  - `dataset_statistics_summary.csv` (mean, std, min, max per metric)
  - `detailed_results.csv` (flattened per-image results)
- PNG plots (300 DPI):
  - `dataset_miou_comparison.png`
  - `dataset_dice_comparison.png`
  - `dataset_boundary_iou_comparison.png`
  - `dataset_boundary_f1_comparison.png`

---

#### Step 5.2: Generate Regional Comparison Plots

Group datasets by geographic region and compare model performance.

```bash
poetry run python report_visualisations/visualise_results_by_region.py
```

**Regional groupings:**
- **Caribbean**: SeaFlower (Bolivar, Courtown), Tetes Providencia, UNAL Tayrona
- **Atlantic (non-Caribbean)**: Seaview Atlantic
- **Indo-Pacific**: Seaview Indonesia/Philippines
- **Pacific – Australia**: Seaview Pacific Australia
- **Pacific – USA**: Seaview Pacific USA

**Output:**
- PNG plots (300 DPI):
  - `regional_miou_comparison.png`
  - `regional_dice_comparison.png`
  - `regional_boundary_iou_comparison.png`
  - `regional_boundary_f1_comparison.png`

---

## Quick Start + Sumbission instructions

### Submission notebook
The notebook `poster_viz.ipynb` contains the submission notebook that should be run as our implementation. It generates all the visualizations used for methodology section components (report and poster). The instructions on how to run it, and given in Installation and Interactive Development sections.

### Final inference pipeline
To process a single underwater image end-to-end:

```bash
poetry run python automated_underwater_area_estimation/main.py path/to/image.jpg
```

**With options:**
```bash
poetry run python automated_underwater_area_estimation/main.py \
  path/to/image.jpg \
  --quadrant-width 54 \
  --quadrant-height 54 \
  --verbose
```

**Output:**
```
Processing image: path/to/image.jpg
Image loading time: 0.15 seconds
Model loading time: 3.42 seconds
Coral segmentation time: 8.76 seconds
Quadrant segmentation time: 1.23 seconds
Area estimation time: 0.08 seconds
Coral pixel count: 1,234,567
Background pixel count: 2,345,678
Coral area (cm²): 1,845.23
Background area (cm²): 3,506.91
```

**Prerequisites:**
- Trained quadrant model must exist at `segmentation_quadrant/segformer_best/`
- EPFL coral model will auto-download from HuggingFace on first run

---

## Development

### Code Formatting

```bash
poetry run black .
```

Auto-excludes: `data/`, `data_preprocessed/`, `__pycache__/`, `.venv/`

### Running Tests

```bash
# All tests
poetry run pytest

# Specific file
poetry run pytest tests/test_segmentation.py

# Specific test
poetry run pytest tests/test_segmentation.py::TestClass::test_method

# Pattern matching
poetry run pytest -k "coral"

# Skip slow tests
poetry run pytest -m "not slow"

# Verbose output
poetry run pytest -v
```

### Adding Dependencies

```bash
# Regular dependency
poetry add package-name

# Development dependency
poetry add --group dev package-name

# Update lock file
poetry lock --no-update
```

### Interactive Development

```bash
# Jupyter notebook
poetry run jupyter notebook

# IPython shell with project imports
poetry run ipython
```

---

## Project Structure

```
automated-underwater-area-estimation/
├── automated_underwater_area_estimation/    # Main package
│   ├── segmentation_corals/                # Coral segmentation models
│   │   ├── model.py                        # Abstract base class
│   │   ├── class_mapping.py                # Class label mappings
│   │   ├── epfl/                           # EPFL Segformer models
│   │   │   ├── model.py                    # EPFLModel implementation
│   │   │   └── classmap.py                 # 39-class CoralScapes mapping
│   │   ├── reefsupport/                    # YOLO models
│   │   │   └── model.py                    # ReefSupportModel
│   │   ├── coralscop/                      # CoralSCOP model
│   │   ├── segmentation_dataset.py         # PyTorch dataset loaders
│   │   ├── evaluation_metrics.py           # Boundary-aware metrics
│   │   └── segmentation_evaluation.py      # Multi-dataset evaluation
│   ├── segmentation_quadrant/              # Quadrant detection
│   │   ├── model.py                        # QuadrantSegmentationModel
│   │   ├── preprocess_raw_masks.py         # Morphological cleaning
│   │   ├── data_augmentation.py            # Albumentations pipeline
│   │   ├── train_model.py                  # Training script
│   │   └── segformer_best/                 # Trained weights (git-ignored)
│   ├── area_estimation/                    # Pixel-to-area conversion
│   │   ├── area_estimation.py              # Corner detection + PAE
│   │   ├── ground_truth_generation.py      # CPC parser
│   │   └── evaluation.py                   # Evaluator class
│   ├── label_studio/                       # Annotation tools
│   │   ├── quadrant_mask_labelling.py      # Streamlit + AquaSAM
│   │   └── aquasam_weights.pth             # SAM weights (download separately)
│   ├── preprocess_data/                    # Dataset preprocessing
│   │   ├── preprocess_IBF.py
│   │   └── preprocess_reefsupport.py
│   ├── data/                               # Raw datasets (git-ignored)
│   ├── data_preprocessed/                  # Processed data (git-ignored)
│   ├── evaluation_results/                 # Model evaluation output
│   ├── main.py                             # CLI for single-image processing
│   ├── utils.py                            # Device detection, helpers
│   └── download_project_data.py            # GCS data downloader
├── report_visualisations/                  # Plotting scripts
│   ├── visualise_results.py                # Dataset-level plots
│   └── visualise_results_by_region.py      # Regional plots
├── tests/                                  # Unit/integration tests
├── pyproject.toml                          # Poetry config + dependencies
├── poetry.lock                             # Locked dependency versions
├── README.md                               # This file
├── WARP.md                                 # Warp AI agent guide
├── LICENSE                                 # MIT License
└── .gitignore                              # Git exclusions
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create a branch** for your feature: `git checkout -b feature-name`
3. **Format code**: `poetry run black .`
4. **Run tests**: `poetry run pytest`
5. **Commit changes**: `git commit -am "Add feature"`
6. **Push to branch**: `git push origin feature-name`
7. **Open a Pull Request**

### Code Style

- Follow Black formatter defaults (88-char lines)
- Use type hints where possible
- Document functions with docstrings (Google style)
- Add tests for new features

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **EPFL-ECEO** for the CoralScapes dataset and pre-trained Segformer models
- **ReefSupport** for YOLO models and underwater datasets
- **Meta AI** for Segment Anything Model (SAM)
- **AquaSAM** team for underwater-fine-tuned SAM weights
- **NVIDIA** for MIT-B0 Segformer architecture
