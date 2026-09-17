# Agent Context & Preferences

This file provides context, user preferences, and tech stack information to guide the AI (Agent) to work more efficiently in this repository.

## Project Context
- **Project**: Kaggle Competition - Digital Academic Wellbeing
- **Goal**: Build a structured, clean, and modular machine learning model (regression/classification) for the Kaggle competition.

## Tech Stack & Core Libraries
- **Package Manager**: **`uv`** (Always use `uv` instead of standard `pip`)
- **Data Manipulation**: `pandas`, `polars` (primary option for large datasets), `numpy`, `pyarrow`, `fastparquet`
- **Machine Learning Models**: `scikit-learn`, `lightgbm`, `xgboost`, `catboost`
- **Hyperparameter Tuning**: `optuna`
- **Experiment Tracking**: `mlflow` or `Weights & Biases (wandb)`
- **Explainability**: `shap`

## Workflow & Coding Preferences
1. **Dependency Management**: DO NOT use `pip install`. Always use `uv` (e.g., `uv pip install <package>` or `uv add <package>` if migrating to `pyproject.toml` later).
2. **Code Modularity**: Keep the core logic inside the `src/` directory (`features.py`, `models.py`, `metrics.py`), while execution scripts should be placed in the `scripts/` directory (`train.py`, `inference.py`).
3. **Data Processing**: Prioritize memory efficiency (e.g., using appropriate data types in `polars` or `pandas`).
4. **Reproducibility**: Ensure *random seeds* are always set consistently at the beginning of each script (`utils.seed_everything`).
5. **Ignore Large Files**: NEVER add files from the `data/` directory (especially `.csv` and `.parquet`) or large model files (`.pkl`, `.joblib`, `.h5`) to the Git history. Always rely on `.gitignore`.

## Phase 1 Conventions & Decisions
During our Phase 1 (Data Forensics & Canonicalization) implementation, we established the following conventions which must be adhered to throughout the project:
1. **Tooling Choice**: While `polars` is recommended for large datasets, we use **`pandas`** for forensics and preprocessing scripts since the dataset is small (4,048 rows), allowing for easier visual exploration and statistical testing.
2. **Handling Impossible Outliers**: Values that are strictly impossible (e.g., negative counts like `notifications = -5`) are converted to `NaN` during the canonicalization step. Do not drop the rows; let the Phase 2 imputer handle them to maintain the dataset distribution.
3. **Canonicalization Output**: Never overwrite raw column data. Always retain the original column and create the cleaned/normalized version with a `_clean` suffix (e.g., `age` -> `age_clean`).
