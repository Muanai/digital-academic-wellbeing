import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import apply_canonicalization
from src.features import generate_features
from src.validation import get_cv_strategy, evaluate_model
from src.models import get_baseline_models, build_baseline_pipeline

def append_to_log(log_path: str, exp_id: str, model: str, features: str, cv_f1_mean: float, notes: str):
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write("exp_id,model,features,cv_f1_mean,notes\n")
    with open(log_path, 'a') as f:
        f.write(f"{exp_id},{model},\"{features}\",{cv_f1_mean:.4f},\"{notes}\"\n")

def main():
    print("Starting Feature Engineering Ablation Study (XGBoost)...")
    data_dir = Path("data/raw")
    train_path = data_dir / "train.csv"
    
    df_train_raw = pd.read_csv(train_path)
    df_train_clean = apply_canonicalization(df_train_raw)
    target_col = 'digital_academic_wellbeing_level'
    target_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df_train_clean[target_col] = df_train_clean[target_col].map(target_map)
    
    # Base XGBoost Model
    models = get_baseline_models(random_state=42)
    xgb_model = models['XGBoost']
    
    mlflow.set_experiment("Phase_3_Feature_Ablation")
    log_path = "experiments/experiment_log.csv"
    os.makedirs("experiments", exist_ok=True)
    
    experiments = [
        ("E006", ["digital"], "XGBoost + Digital"),
        ("E007", ["academic"], "XGBoost + Academic"),
        ("E008", ["cross_domain"], "XGBoost + Cross-Domain"),
        ("E009", ["aggregate"], "XGBoost + Aggregate"),
        ("E010", ["binning"], "XGBoost + Binning"),
        ("E011", ["digital", "academic", "cross_domain", "aggregate", "binning"], "XGBoost + Best Combo (All)")
    ]
    
    cv = get_cv_strategy(n_splits=5, random_state=42)
    y = df_train_clean[target_col]
    
    for exp_id, families, notes in experiments:
        print(f"\nRunning {exp_id}: {notes}")
        
        # 1. Generate features for this specific experiment
        df_feat = generate_features(df_train_clean, families=families)
        
        # 2. Cleanup columns
        cols_to_drop = ['respondent_id', target_col]
        raw_to_drop = [c.replace('_clean', '') for c in df_feat.columns if c.endswith('_clean')]
        cols_to_drop.extend(raw_to_drop)
        X = df_feat.drop(columns=cols_to_drop, errors='ignore')
        
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        pipeline = build_baseline_pipeline(xgb_model, numeric_features, categorical_features)
        
        with mlflow.start_run(run_name=f"{exp_id}_XGBoost"):
            mlflow.log_param("families", "+".join(families))
            mean_f1, oof_preds = evaluate_model(pipeline, X, y, cv)
            
            mlflow.log_metric("macro_f1_oof", mean_f1)
            
            append_to_log(log_path, exp_id, "XGBoost", "+".join(families), mean_f1, notes)
            print(f"  {exp_id} Macro-F1: {mean_f1:.4f}")

if __name__ == "__main__":
    main()
