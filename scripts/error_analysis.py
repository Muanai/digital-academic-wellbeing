import os
import sys
import pandas as pd
import numpy as np
import mlflow
import ast
from pathlib import Path
from sklearn.metrics import classification_report

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

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def get_best_params_from_mlflow(experiment_name: str, model_name: str):
    """Fetch best params from MLflow tracking server for a given model."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment {experiment_name} not found.")
        
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"params.model = '{model_name}'"
    )
    
    if runs.empty:
        raise ValueError(f"No runs found for model {model_name}")
        
    # Get the run with the best tuned macro-f1
    best_run = runs.sort_values(by="metrics.best_tuned_macro_f1", ascending=False).iloc[0]
    
    # Extract all params starting with 'params.best_'
    params = {}
    for col in best_run.index:
        if col.startswith('params.best_'):
            params[col] = best_run[col]
            
    return params

def get_model_from_params(model_name: str, params: dict, random_state: int = 42):
    """Reconstructs the model using string params fetched from MLflow."""
    if model_name == "LogisticRegression":
        return LogisticRegression(
            class_weight='balanced', 
            C=float(params["params.best_C"]), 
            random_state=random_state, 
            max_iter=1000
        )
        
    elif model_name == "RandomForest":
        return RandomForestClassifier(
            class_weight='balanced', 
            n_estimators=int(params["params.best_n_estimators"]), 
            max_depth=int(params["params.best_max_depth"]), 
            min_samples_split=int(params["params.best_min_samples_split"]), 
            random_state=random_state, n_jobs=-1
        )
        
    elif model_name == "ExtraTrees":
        return ExtraTreesClassifier(
            class_weight='balanced', 
            n_estimators=int(params["params.best_n_estimators"]), 
            max_depth=int(params["params.best_max_depth"]),
            min_samples_split=int(params["params.best_min_samples_split"]), 
            random_state=random_state, n_jobs=-1
        )
        
    elif model_name == "MLPClassifier":
        # Parse tuple from string (e.g. "(64, 32)")
        hidden_layer_sizes = ast.literal_eval(params["params.best_hidden_layer_sizes"])
        return MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes, 
            alpha=float(params["params.best_alpha"]), 
            learning_rate_init=float(params["params.best_learning_rate_init"]),
            max_iter=1000, early_stopping=True, random_state=random_state
        )
        
    elif model_name == "LightGBM":
        return lgb.LGBMClassifier(
            class_weight='balanced', 
            learning_rate=float(params["params.best_learning_rate"]), 
            num_leaves=int(params["params.best_num_leaves"]),
            max_depth=int(params["params.best_max_depth"]), 
            min_child_samples=int(params["params.best_min_child_samples"]), 
            subsample=float(params["params.best_subsample"]),
            colsample_bytree=float(params["params.best_colsample_bytree"]), 
            n_estimators=100, random_state=random_state, verbose=-1
        )
        
    elif model_name == "XGBoost":
        return xgb.XGBClassifier(
            learning_rate=float(params["params.best_learning_rate"]), 
            max_depth=int(params["params.best_max_depth"]), 
            min_child_weight=int(params["params.best_min_child_weight"]),
            subsample=float(params["params.best_subsample"]), 
            colsample_bytree=float(params["params.best_colsample_bytree"]), 
            n_estimators=100,
            random_state=random_state, eval_metric='mlogloss', n_jobs=-1
        )
        
    elif model_name == "CatBoost":
        return CatBoostClassifier(
            auto_class_weights='Balanced', 
            learning_rate=float(params["params.best_learning_rate"]), 
            depth=int(params["params.best_depth"]),
            l2_leaf_reg=float(params["params.best_l2_leaf_reg"]), 
            iterations=100, random_state=random_state, verbose=False
        )
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def main():
    print("Starting Phase 5: Error Analysis...")
    data_dir = Path("data/raw")
    train_path = data_dir / "train.csv"
    
    if not train_path.exists():
        print(f"Error: Data not found at {train_path}")
        return
        
    df_train_raw = pd.read_csv(train_path)
    
    # 1. Prepare Data
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
    
    cv = get_cv_strategy(n_splits=5, random_state=42)
    xgb_sample_weight = compute_xgb_sample_weights(y)
    
    # Point to local sqlite mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    models_to_analyze = [
        "CatBoost", 
        "XGBoost", 
        "LightGBM", 
        "RandomForest",
        "LogisticRegression", 
        "MLPClassifier",
        "ExtraTrees"
    ]
    
    artifact_dir = "experiments/phase5_artifacts"
    os.makedirs(artifact_dir, exist_ok=True)
    
    # We will build error_df with all predictions
    error_df = pd.DataFrame({'y_true': y})
    oof_probas = {}
    
    print("\n" + "="*50)
    print("RECONSTRUCTING MODELS AND GENERATING PREDICTIONS")
    print("="*50)
    
    for model_name in models_to_analyze:
        print(f"\nProcessing {model_name}...")
        
        # 1. Fetch best params
        params = get_best_params_from_mlflow("Phase_4_Optuna_Tuning", model_name)
        model = get_model_from_params(model_name, params)
        pipeline = build_baseline_pipeline(model, numeric_features, categorical_features)
        
        # 2. Run CV
        sw = xgb_sample_weight if model_name == "XGBoost" else None
        
        # Mute evaluate_model printing
        import builtins
        original_print = builtins.print
        builtins.print = lambda *args, **kwargs: None
        _, oof_preds, oof_proba = evaluate_model(pipeline, X, y, cv, sample_weight=sw)
        builtins.print = original_print
        
        # 3. Optimize thresholds
        best_f1, best_weights, best_preds = optimize_thresholds(y, oof_proba)
        
        # 4. Save results
        oof_probas[model_name] = oof_proba
        error_df[f"{model_name}_pred"] = best_preds
        error_df[f"{model_name}_wrong"] = best_preds != y
        
        # 5. Print Classification Report
        target_names = ['Low', 'Medium', 'High']
        print(f"Classification Report for {model_name} (Tuned Macro-F1: {best_f1:.4f}):")
        print(classification_report(y, best_preds, target_names=target_names))
        
        # Save OOF Proba to disk for Phase 6
        proba_path = os.path.join(artifact_dir, f"{model_name}_oof_proba.npy")
        np.save(proba_path, oof_proba)
    
    # Save Error DF
    error_df_path = os.path.join(artifact_dir, "error_df.csv")
    error_df.to_csv(error_df_path, index=False)
    
    # 6. Analyze Model Disagreements
    print("\n" + "="*50)
    print("ERROR CORRELATION ANALYSIS (Are models making the same mistakes?)")
    print("="*50)
    
    wrong_cols = [f"{m}_wrong" for m in models_to_analyze]
    error_corr = error_df[wrong_cols].corr()
    
    print("\nError Correlation Matrix (Lower is better for ensembling):")
    # Simplify column names for display
    display_corr = error_corr.copy()
    display_corr.columns = models_to_analyze
    display_corr.index = models_to_analyze
    print(display_corr.round(3))
    
    print("\n" + "="*50)
    print("BOTTLENECK ANALYSIS: THE 'HIGH' CLASS")
    print("="*50)
    
    high_mask = error_df['y_true'] == 2 # 2 is 'High'
    total_high = high_mask.sum()
    
    for model_name in models_to_analyze:
        wrong_count = error_df.loc[high_mask, f"{model_name}_wrong"].sum()
        print(f"{model_name}: Missed {wrong_count}/{total_high} 'High' instances ({(wrong_count/total_high)*100:.1f}%)")
        
    print(f"\nArtifacts saved to {artifact_dir}")

if __name__ == "__main__":
    main()
