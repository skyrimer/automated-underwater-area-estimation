# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a Python project for automated underwater area estimation using machine learning segmentation models. The project implements two different segmentation approaches: EPFL Segformer models for coral reef analysis and ReefSupport YOLO models for underwater object detection.

## Development Commands

### Environment Setup
```bash
# Install dependencies (Poetry manages both regular and dev dependencies)
poetry install

# Activate environment
poetry shell

# Or run commands without activating shell
poetry run <command>
```

### Development Workflow
```bash
# Format code with Black
poetry run black .

# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_segmentation.py

# Run tests with verbose output
poetry run pytest -v

# Run Jupyter notebook for experimentation
poetry run jupyter notebook
```

### Single Test Execution
```bash
# Run a specific test method
poetry run pytest tests/test_segmentation.py::TestSegmentationModels::test_epfl_model_loading_b2

# Run tests matching a pattern
poetry run pytest -k "epfl_model"

# Skip slow tests
poetry run pytest -m "not slow"
```

## Architecture Overview

### Core Design Pattern
The project uses an abstract base class architecture centered around `SegmentationModelBase` that enforces a consistent interface across different model implementations.

**Key architectural components:**

1. **Abstract Base Classes**: 
   - `SegmentationModelBase`: Defines the contract for all segmentation models
   - `ClassMappingBase`: Ensures consistent class mapping across datasets

2. **Model Implementations**:
   - `EPFLModel`: Wraps HuggingFace Segformer models for coral reef segmentation (1024x1024 input)
   - `ReefSupportModel`: Wraps YOLO models for underwater object detection (640x640 input)

3. **Dataset-Specific Class Mappings**:
   - `EPFLClassMapping`: 39 classes for coral reef analysis (seagrass, coral types, fish, etc.)
   - `ReefSupportClassMapping`: Classes for YOLO-based detection

### Model Factory Pattern
Models are instantiated with specific pretrained weights:
- **EPFL models**: `segformer-b2` and `segformer-b5` variants from HuggingFace
- **ReefSupport models**: `yolov8_sm_latest.pt` and `yolov8_xlarge_latest.pt` with automatic download from HuggingFace

### Device Management
All models implement automatic device detection with fallback priority:
1. CUDA (NVIDIA GPUs)
2. MPS (Apple Silicon)
3. XPU (Intel GPUs) 
4. CPU

Access via `get_best_device()` utility function.

### Data Pipeline Architecture
- **Input**: PIL Images
- **Preprocessing**: Model-specific (HuggingFace transformers vs YOLO native)
- **Output**: Tuple of (processed_image, segmentation_tensor)
- **Postprocessing**: Class ID to name mapping via dataset-specific mappings

## Key Implementation Details

### Model Loading and Validation
All models inherit validation that ensures required attributes are present:
- `model_name`, `preprocessor`, `model`, `class_mapping`, `ideal_size`
- Type validation for `ideal_size` (tuple of positive integers)
- Interface validation for `class_mapping` (must inherit from `ClassMappingBase`)

### File Download System
ReefSupport models implement automatic model weight downloading:
- Downloads to `automated_underwater_area_estimation/segmentation/reefsupport/models/`
- Uses torch.hub for reliable downloads
- Checks for existing files to avoid re-downloading

### Testing Architecture
Comprehensive test suite using pytest with:
- **Shared fixtures**: CoralScapes dataset loading in `setup_class`
- **Parameterized tests**: Testing both model variants
- **Integration tests**: Full pipeline from image to segmentation
- **Validation tests**: Model attribute and interface checking

### Data Access
The project includes GCS bucket download functionality for accessing coral reef datasets, particularly the CoralScapes dataset used for training and evaluation.

## Python Version Requirements
- **Required**: Python 3.12 or higher (but less than 3.14)
- Dependencies managed through Poetry with lock file for reproducible builds

## Dataset Context
- **Primary**: CoralScapes dataset from EPFL-ECEO for coral reef analysis
- **Secondary**: ReefSupport datasets for YOLO-based underwater detection
- **Test data**: Uses CoralScapes test split (index 42 for reproducibility)

## Development Notes
- Black formatter configuration excludes `automated_underwater_area_estimation/data` directory
- Test configuration includes slow test markers for computational intensive tests
- Jupyter notebooks are supported for experimentation (`test.ipynb` demonstrates usage)