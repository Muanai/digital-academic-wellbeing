import sys
import warnings
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.preprocessing import apply_canonicalization
from src.features import get_model_specific_features
from src.validation import get_cv_strategy, evaluate_model, optimize_thresholds
from src.models import get_baseline_models, build_baseline_pipeline
from src.utils import seed_everything

def run_phase14():
    seed_everything(42)
    print("Loading data...")
    df_raw = pd.read_csv('data/raw/train.csv')
    
    print("Applying base preprocessing...")
    df = apply_canonicalization(df_raw)
    
    target_col = 'digital_academic_wellbeing_level'
    cols_to_drop = ['respondent_id', target_col]
    raw_to_drop = [c.replace('_clean', '') for c in df.columns if c.endswith('_clean')]
    cols_to_drop.extend(raw_to_drop)
    y = df[target_col].map({'Low': 0, 'Medium': 1, 'High': 2})
    
    cv = get_cv_strategy(n_splits=5, random_state=42)
    
    # 1. Load Baseline Models (Golden Scenario components)
    golden_models = get_baseline_models(random_state=42)
    # Filter to only the 5 main ones from Golden Scenario (Scenario 4 / Scenario 2 equivalent)
    target_models = {
        "CatBoost": golden_models["CatBoost_s1"],
        "XGBoost": golden_models["XGBoost_s1"],
        "LightGBM": golden_models["LightGBM_s1"],
        "RandomForest": golden_models["RandomForest"],
        "ExtraTrees": golden_models["ExtraTrees"]
    }
    
    oof_probas_dict = {}
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Phase15_Screentime_Fix")
    
    with mlflow.start_run(run_name="Phase15_Dynamic_Pipeline_Clipped"):
        print("\n--- Evaluating Models with New Pipelines ---")
        for model_name, model in target_models.items():
            print(f"  > Generating OOF for {model_name}...")
            # We use tree/linear ablated features as done in Golden Scenario
            model_type = "tree" if any(t in model_name for t in ["CatBoost", "XGBoost", "LightGBM", "RandomForest", "ExtraTrees"]) else "linear"
            
            df_feat = get_model_specific_features(df, model_type=model_type)
            X = df_feat.drop(columns=cols_to_drop, errors='ignore')
            numeric_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
            categorical_features = [c for c in X.columns if c not in numeric_features]
            
            pipeline = build_baseline_pipeline(model, numeric_features, categorical_features)
            
            if "XGBoost" in model_name:
                from src.models import compute_xgb_sample_weights
                sw = compute_xgb_sample_weights(y)
                score, _, oof_proba = evaluate_model(pipeline, X, y, cv, sample_weight=sw)
            else:
                score, _, oof_proba = evaluate_model(pipeline, X, y, cv)
                
            oof_probas_dict[model_name] = oof_proba
            
            print(f"  Overall OOF Macro-F1: {score:.4f}")
            mlflow.log_metric(f"{model_name}_cv_macro_f1", score)
            
        # Golden Scenario Soft Voting (Equal Weights to start, then optimize)
        print("\n--- Golden Scenario Soft-Voting Ensemble (Equal Weights) ---")
        ensemble_proba = np.zeros((len(df), 3))
        for proba in oof_probas_dict.values():
            ensemble_proba += proba
        ensemble_proba /= len(target_models)
        
        ensemble_f1, best_thresholds, _ = optimize_thresholds(y, ensemble_proba)
        
        y_pred_argmax = np.argmax(ensemble_proba, axis=1)
        ensemble_argmax_f1 = f1_score(y, y_pred_argmax, average='macro')
        
        print(f"Phase 14 Ensemble (Argmax) Macro-F1    : {ensemble_argmax_f1:.4f}")
        print(f"Phase 14 Ensemble (Optimized) Macro-F1 : {ensemble_f1:.4f}")
        
        mlflow.log_metric("ensemble_argmax_macro_f1", ensemble_argmax_f1)
        mlflow.log_metric("ensemble_optimized_macro_f1", ensemble_f1)
        
if __name__ == '__main__':
    run_phase14()
