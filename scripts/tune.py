import os
import sys
import pandas as pd
import numpy as np
import mlflow
import optuna
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import apply_canonicalization
from src.features import generate_features
from src.validation import get_cv_strategy, evaluate_model, optimize_thresholds
from src.models import build_baseline_pipeline, compute_xgb_sample_weights
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Filter out convergence warnings from LogisticRegression/MLP for cleaner logs
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def get_model_from_trial(trial, model_name: str, random_state: int = 42):
    """
    Returns an instantiated model with hyperparameters suggested by Optuna.
    """
    if model_name == "LogisticRegression":
        C = trial.suggest_float("C", 1e-4, 10.0, log=True)
        return LogisticRegression(class_weight='balanced', C=C, random_state=random_state, max_iter=1000)
        
    elif model_name == "RandomForest":
        n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        return RandomForestClassifier(
            class_weight='balanced', n_estimators=n_estimators, max_depth=max_depth, 
            min_samples_split=min_samples_split, random_state=random_state, n_jobs=-1
        )
        
    elif model_name == "ExtraTrees":
        n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        return ExtraTreesClassifier(
            class_weight='balanced', n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, random_state=random_state, n_jobs=-1
        )
        
    elif model_name == "MLPClassifier":
        hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(64, 32), (128, 64), (64,), (32,)])
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True)
        return MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, learning_rate_init=learning_rate_init,
            max_iter=1000, early_stopping=True, random_state=random_state
        )
        
    elif model_name == "LightGBM":
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        num_leaves = trial.suggest_int("num_leaves", 10, 100)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        min_child_samples = trial.suggest_int("min_child_samples", 5, 50)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        return lgb.LGBMClassifier(
            class_weight='balanced', learning_rate=learning_rate, num_leaves=num_leaves,
            max_depth=max_depth, min_child_samples=min_child_samples, subsample=subsample,
            colsample_bytree=colsample_bytree, n_estimators=100, random_state=random_state, verbose=-1
        )
        
    elif model_name == "XGBoost":
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        return xgb.XGBClassifier(
            learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight,
            subsample=subsample, colsample_bytree=colsample_bytree, n_estimators=100,
            random_state=random_state, eval_metric='mlogloss', n_jobs=-1
        )
        
    elif model_name == "CatBoost":
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        depth = trial.suggest_int("depth", 3, 10)
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 10.0)
        return CatBoostClassifier(
            auto_class_weights='Balanced', learning_rate=learning_rate, depth=depth,
            l2_leaf_reg=l2_leaf_reg, iterations=100, random_state=random_state, verbose=False
        )
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def main():
    print("Starting Phase 4 Hyperparameter Tuning (Optuna)...")
    data_dir = Path("data/raw")
    train_path = data_dir / "train.csv"
    
    if not train_path.exists():
        print(f"Error: Data not found at {train_path}")
        return
        
    df_train_raw = pd.read_csv(train_path)
    
    # 1. Apply Canonicalization & Feature Engineering
    df_train_clean = apply_canonicalization(df_train_raw)
    df_train_engineered = generate_features(df_train_clean)
    
    target_col = 'digital_academic_wellbeing_level'
    target_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df_train_engineered[target_col] = df_train_engineered[target_col].map(target_map)
    
    cols_to_drop = ['respondent_id', target_col]
    raw_to_drop = [c.replace('_clean', '') for c in df_train_engineered.columns if c.endswith('_clean')]
    cols_to_drop.extend(raw_to_drop)
    
    X = df_train_engineered.drop(columns=cols_to_drop, errors='ignore')
    y = df_train_engineered[target_col]
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 2. Setup CV
    cv = get_cv_strategy(n_splits=5, random_state=42)
    xgb_sample_weight = compute_xgb_sample_weights(y)
    
    # Models to tune
    models_to_tune = [
        "LightGBM", 
        "LogisticRegression", 
        "XGBoost", 
        "RandomForest", 
        "ExtraTrees",
        "CatBoost",
        "MLPClassifier"
    ]
    
    n_trials = 50
    mlflow.set_experiment("Phase_4_Optuna_Tuning")
    
    # We suppress evaluate_model prints during optuna tuning so the console is readable
    import builtins
    original_print = builtins.print
    
    for model_name in models_to_tune:
        print(f"\n{'='*50}\nStarting Optuna Study for {model_name}\n{'='*50}")
        
        with mlflow.start_run(run_name=f"Tuning_{model_name}"):
            mlflow.log_param("model", model_name)
            mlflow.log_param("n_trials", n_trials)
            
            def objective(trial):
                # Instantiate model with trial parameters
                model = get_model_from_trial(trial, model_name)
                pipeline = build_baseline_pipeline(model, numeric_features, categorical_features)
                
                # Mute prints inside evaluate_model
                builtins.print = lambda *args, **kwargs: None
                
                sw = xgb_sample_weight if model_name == "XGBoost" else None
                overall_score, oof_preds, oof_proba = evaluate_model(pipeline, X, y, cv, sample_weight=sw)
                
                builtins.print = original_print
                
                # Optuna metric is the Tuned Macro-F1 (with threshold optimization)
                best_f1, best_weights, best_preds = optimize_thresholds(y, oof_proba)
                
                # If model doesn't support predict_proba, fallback to default score
                final_score = best_f1 if best_f1 is not None else overall_score
                
                return final_score
            
            # Use TPE Sampler with a fixed seed for reproducibility
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction="maximize", sampler=sampler, study_name=f"study_{model_name}")
            study.optimize(objective, n_trials=n_trials, n_jobs=1)  # n_jobs=1 because of nested multiprocessing in tree models
            
            print(f"\nBest trials for {model_name}:")
            print(f"  Value (Tuned Macro-F1): {study.best_value:.4f}")
            print(f"  Params: {study.best_params}")
            
            # Log best params to MLflow
            mlflow.log_metric("best_tuned_macro_f1", study.best_value)
            for k, v in study.best_params.items():
                mlflow.log_param(f"best_{k}", v)

    print("\nPhase 4 Tuning Complete. All studies logged to MLflow.")

if __name__ == "__main__":
    main()
