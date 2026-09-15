# Digital Academic Wellbeing - Kaggle Competition

This repository contains the codebase for the Digital Academic Wellbeing Kaggle competition.

## Directory Structure

- `data/`: Stores raw data, processed data, and submissions. *(Ignored from Git)*
- `src/`: Functional modules for data processing, feature engineering, models, metrics, etc.
- `scripts/`: Main entry points for training and inference.
- `notebooks/`: Jupyter Notebooks for Exploratory Data Analysis (EDA).
- `configs/`: Training configuration files (YAML).
- `experiments/`: Model logs and artifacts.

## Usage

1. **Environment Setup** (using `uv`):
   ```bash
   # This will automatically create a .venv and install all dependencies from pyproject.toml
   uv sync
   ```
2. **Download Data**: Place the Kaggle competition data files into the `data/raw/` directory.
3. **Training**:
   ```bash
   uv run python scripts/train.py --config configs/default.yaml
   ```
4. **Inference (Submission)**:
   ```bash
   uv run python scripts/inference.py --config configs/default.yaml
   ```

## Tech Stack
- Pandas, Polars, Numpy
- Scikit-learn, XGBoost, LightGBM, CatBoost
- Optuna (Tuning), SHAP (Explainability)
- MLflow / Weights & Biases (Experiment Tracking)
