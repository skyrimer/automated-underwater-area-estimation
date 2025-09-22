# Automated Underwater Area Estimation

A Python project for underwater area estimation using machine learning segmentation models.

---

## Prerequisites

### Supported Python Version
This project requires **Python 3.12 or higher (but less than 3.14)**. To check your current Python version, run the following command:
```
bash
python --version
```
Or if your system uses `python3`:
```
bash
python3 --version
```
---

## Setting Up the Project

### 1. Clone the Repository
Start by cloning the repository to your local machine:
```
bash
git clone https://github.com/yourusername/automated-underwater-area-estimation.git
cd automated-underwater-area-estimation
```
### 2. Install Dependencies with Poetry
This project uses `Poetry` to manage dependencies. If you don't have `Poetry` installed, you can install it by following the official guide: [Install Poetry](https://python-poetry.org/docs/#installation)

Once `Poetry` is installed, run the following commands to install the required dependencies:
```
bash
poetry install
```
This will install both regular and development dependencies such as `pytest` and `black`.

### 3. Activate the Poetry Environment
After installation, activate the environment created by `Poetry`:
```
bash
poetry shell
```
This step is optional—alternatively, you can execute commands directly through `Poetry` without activating the shell by prefixing them with `poetry run`.

---

## Adding New Dependencies

To add a new dependency to the project:

1. **For regular dependencies:**

   ```bash
   poetry add <package_name>
   ```

   For example, to add `numpy`:

   ```bash
   poetry add numpy
   ```

2. **For development-only dependencies:**

   ```bash
   poetry add --group dev <package_name>
   ```

   For example, to add `flake8` as a development dependency:

   ```bash
   poetry add --group dev flake8
   ```

This will update the `poetry.lock` and `pyproject.toml` files automatically.

---

## Running the Project

To run the project or test models, simply run the required scripts or Jupyter notebooks (e.g., `test.ipynb`). Use `jupyter` to interactively execute cells.

Start Jupyter Notebook with:
```
bash
poetry run jupyter notebook
```
Or if you'd like to execute a specific script:
```
bash
poetry run python <script_name>.py
```
---

## Pre-Submission Checks

Before submitting any code or contributions, perform the following checks:

### 1. Code Formatting with Black
Ensure the code is properly formatted to maintain consistency. This project uses the `black` formatter. Run the following command to format all Python files:
```
bash
poetry run black .
```
### 2. Running Tests with Pytest
Run all the test cases specified in the `tests` directory using `pytest`:
```
bash
poetry run pytest
```
**Common options for pytest:**
- Run a specific test file:

  ```bash
  poetry run pytest tests/test_specific_file.py
  ```

- Run tests with verbose output:

  ```bash
  poetry run pytest -v
  ```

- Skip tests with a specific marker:

  ```bash
  poetry run pytest -m "not slow"
  ```

Refer to `pyproject.toml` for test configuration options.

---

## Project Structure
```
plaintext
automated-underwater-area-estimation/
├── automated_underwater_area_estimation/   # Main project code
│   └── segmentation/                      # Segmentation models
├── tests/                                 # Unit and integration tests
├── .gitignore                             # Ignored files/directories
├── LICENSE                                # Project license file
├── poetry.lock                            # Poetry lock file
├── pyproject.toml                         # Configuration for Poetry & Pytest
├── README.md                              # This README file
└── test.ipynb                             # Example Jupyter notebook
```
---

## Issues and Contributions

If you encounter a problem or have suggestions for new features, feel free to create an issue in this repository. Contributions are welcome via pull requests, but make sure all tests pass and code is formatted before submitting.

### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.