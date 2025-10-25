# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Automated underwater area estimation system using deep learning segmentation models. The project performs two-stage analysis: (1) coral segmentation using EPFL Segformer models, and (2) quadrant detection for pixel-to-area conversion, enabling precise coral coverage measurements in cm².

## Development Commands

### Environment Setup
```bash
# Install dependencies (includes dev tools: black, pytest, jupyter, streamlit)
poetry install

# On Windows with CUDA support (adjust CUDA version as needed)
poetry source add --priority=explicit pytorch-cu129 https://download.pytorch.org/whl/cu129
poetry add --group cuda --source pytorch-cu129 torch torchvision
poetry install --with cuda

# Activate environment
poetry shell

# Or run commands without activating shell
poetry run <command>
```

### Core Workflow
```bash
# Format code with Black (auto-excludes data directories)
poetry run black .

# Run all tests with verbose output
poetry run pytest

# Run specific test file
poetry run pytest tests/test_specific.py

# Run specific test method
poetry run pytest tests/test_file.py::TestClass::test_method

# Run tests matching a pattern
poetry run pytest -k "pattern"

# Skip slow tests
poetry run pytest -m "not slow"

# Launch Jupyter for experimentation
poetry run jupyter notebook

# Run Streamlit apps (for interactive labeling/visualization)
poetry run streamlit run <script.py>
```

### Main Pipeline Execution
```bash
# Process a single underwater image
poetry run python automated_underwater_area_estimation/main.py <image_path>

# With custom quadrant dimensions and verbose output
poetry run python automated_underwater_area_estimation/main.py <image_path> --quadrant-width 54 --quadrant-height 54 --verbose
```

## Architecture Overview

### Two-Stage Processing Pipeline

The system operates in two independent stages:

1. **Coral Segmentation** (`segmentation_corals/`)
   - Abstract base: `SegmentationModelBase` defines interface for all coral segmentation models
   - Implementation: `EPFLModel` wraps HuggingFace Segformer models (b2/b5 variants)
   - Input: High-resolution underwater images (any size)
   - Output: Binary mask identifying coral vs non-coral pixels
   - Key feature: Sliding window inference for arbitrary image sizes (1024x1024 windows with 1.5x overlap factor)

2. **Quadrant Detection** (`segmentation_quadrant/`)
   - Model: `QuadrantSegmentationModel` (fine-tuned Segformer)
   - Input: Same underwater image
   - Output: Binary mask of the sampling quadrant (reference frame)
   - Preprocessing: Resizes to 800x600 during inference, then scales back to original size
   - Purpose: Identifies the physical reference frame for pixel-to-cm² conversion

3. **Area Estimation** (`area_estimation/`)
   - Combines both masks to compute coral coverage in cm²
   - Algorithm: Extracts quadrant corners (TL/TR/BR/BL), computes pixel-to-cm² ratio via median filtering
   - Output: PAE (Projected Area Estimate) - area per pixel in cm²

### Class Mapping Architecture

- `ClassMappingBase`: Abstract base enforcing validation rules (no duplicates, non-empty names)
- `EPFLClassMapping`: 39 coral reef classes (seagrass, hard coral, soft coral, fish, sand, etc.)
- Validation at initialization prevents invalid mappings from propagating through the system

### Device Management Strategy

All models use `get_best_device()` utility with priority:
1. CUDA (NVIDIA) - prints GPU name and memory
2. MPS (Apple Silicon)
3. XPU (Intel)
4. CPU - prints thread count

Force specific device via `force_device` parameter when needed.

### Sliding Window Inference (EPFLModel)

High-resolution images are processed using overlapping windows:
- Grid calculation: `h_grids = int(np.round(1.5 * h_img / h_crop))`
- Stride computation ensures complete coverage
- Logits accumulated with overlap counting
- Final prediction: averaged across overlapping regions
- No zero-coverage pixels allowed (assertion check)

## Key Implementation Details

### Model Validation Contract

All `SegmentationModelBase` subclasses must define:
- `model_name` (str): HuggingFace identifier or local path
- `preprocessor`: Transforms PIL images to model inputs
- `model`: Actual torch model
- `class_mapping` (ClassMappingBase): Dataset-specific class definitions
- `ideal_size` (Tuple[int, int]): Optimal input dimensions

Validation occurs in `__init__`, failing fast on missing/invalid attributes.

### Quadrant Corner Detection

Algorithm in `area_estimation.py`:
- **Top-Left**: Minimize (x+y) with x>0, y>0 constraint
- **Top-Right**: Maximize (x-y), tie-break on largest x then smallest y
- **Bottom-Left**: Maximize (y-x), tie-break on largest y then smallest x  
- **Bottom-Right**: Maximize x*y

Computes 6 distances (4 edges + 2 diagonals), filters via median ±8% tolerance, returns averaged PAE.

### Data Download System

`download_project_data.py` provides GCS bucket access:
- Used for CoralScapes dataset and training data
- Manual execution required (not automated in pipeline)

### Testing Configuration

From `pyproject.toml`:
- Test discovery: `tests/test_*.py`
- Default options: `-v --tb=short --strict-markers`
- Markers: `slow` for computationally intensive tests
- Warnings suppressed: DeprecationWarning, PendingDeprecationWarning

### Label Studio Integration

`label_studio/quadrant_mask_labelling.py`: Streamlit app for manual quadrant annotation
- Uses Segment Anything Model (SAM) for interactive segmentation
- Coordinates-based point selection for mask refinement

## Data Preprocessing

### Input Requirements

- **Coral Segmentation**: RGB PIL images, any resolution (recommended >1024px on smaller side)
- **Quadrant Detection**: Same images, internally resized to 800x600 for model input

### Directory Structure

```
automated_underwater_area_estimation/
├── segmentation_corals/          # Coral segmentation models
│   ├── model.py                  # Abstract base class
│   ├── class_mapping.py          # Class mapping validation
│   ├── epfl/                     # EPFL Segformer implementation
│   ├── reefsupport/              # YOLO-based detection
│   └── coralscop/                # Alternative model implementations
├── segmentation_quadrant/        # Quadrant detection
│   ├── model.py                  # QuadrantSegmentationModel
│   ├── train_model.py            # Training script
│   ├── data_augmentation.py      # Augmentation pipeline
│   └── segformer_best/           # Trained model weights (required)
├── area_estimation/              # Pixel-to-area conversion
│   ├── area_estimation.py        # Corner detection + PAE computation
│   ├── evaluation.py             # Evaluator class for metrics
│   └── ground_truth_generation.py
├── preprocess_data/              # Dataset preprocessing
│   ├── preprocess_IBF.py
│   └── preprocess_reefsupport.py
├── label_studio/                 # Manual annotation tools
├── main.py                       # End-to-end pipeline CLI
└── utils.py                      # Device detection, plotting utilities
```

## Python Environment

- **Required**: Python 3.12 (not 3.13+)
- **Lock file**: `poetry.lock` ensures reproducible builds
- **Key dependencies**: torch, transformers, ultralytics, opencv, scikit-image, albumentationsx
- **Dev dependencies**: black, pytest, jupyter, streamlit, segment-anything

## Important Notes

### Black Formatting Exclusions

Automatically skips:
- `automated_underwater_area_estimation/data/`
- `automated_underwater_area_estimation/data_preprocessed/`

### Model Weights Location

- **Coral models**: Downloaded from HuggingFace on first use (EPFL-ECEO repos)
- **Quadrant model**: Must exist at `segmentation_quadrant/segformer_best/` (trained locally)
- **YOLO models**: Auto-download to `segmentation_corals/reefsupport/models/`

### Evaluation Metrics

When testing area estimation accuracy:
- MAE (Mean Absolute Error in cm²)
- RMSE (Root Mean Squared Error)
- R² (coefficient of determination)
- Relative error (fraction and percentage)
- Bias (systematic over/under-estimation)

See `Evaluator` class in `area_estimation/evaluation.py` for CSV-based batch evaluation.
