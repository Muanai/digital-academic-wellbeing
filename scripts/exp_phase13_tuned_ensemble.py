import sys
import json
import warnings
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.validation import get_cv_strategy, optimize_thresholds
from src.utils import seed_everything
from scripts.exp_phase12_hyperparameter_tuning import precompute_folds

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

def load_best_params():
    try:
        with open('experiments/phase12_best_params.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: phase12_best_params.json not found! Run tuning first.")
        sys.exit(1)

def run_tuned_ensemble():
    seed_everything(42)
    print("Loading data and tuned hyperparameters...")
    df_raw = pd.read_csv('data/raw/train.csv')
    best_params = load_best_params()
    
    cv = get_cv_strategy(n_splits=5, random_state=42)
    target_col = 'digital_academic_wellbeing_level'
    label_map = {'Low': 0, 'Medium': 1, 'High': 2}
    
    # Use exact same fold precomputation logic as Phase 12 tuning
    # This ensures feature generation is identical and prevents data leakage
    folds_data = precompute_folds(df_raw, target_col, label_map, cv)
    len_df = len(df_raw)
    
    # Initialize tuned models
    models = {
        "LightGBM_Tuned": lgb.LGBMClassifier(
            **best_params['LightGBM']['params'],
            class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1
        ),
        "XGBoost_Tuned": xgb.XGBClassifier(
            **best_params['XGBoost']['params'],
            random_state=42, eval_metric='mlogloss', n_jobs=-1
        ),
        "CatBoost_Tuned": CatBoostClassifier(
            **best_params['CatBoost']['params'],
            auto_class_weights='Balanced', random_state=42, verbose=False, thread_count=-1
        )
    }
    
    oof_probas_dict = {name: np.zeros((len_df, 3)) for name in models.keys()}
    y_true_all = np.zeros(len_df)
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Phase13_Tuned_Ensemble")
    
    with mlflow.start_run(run_name="Tuned_Golden_Ensemble"):
        for model_name, model in models.items():
            print(f"\n--- Evaluating {model_name} ---")
            
            for fold_data in folds_data:
                # Fit
                if "XGBoost" in model_name:
                    model.fit(fold_data['X_train'], fold_data['y_train'], sample_weight=fold_data['sw_train'])
                else:
                    model.fit(fold_data['X_train'], fold_data['y_train'])
                
                # Predict OOF
                proba = model.predict_proba(fold_data['X_val'])
                oof_probas_dict[model_name][fold_data['val_idx']] = proba
                y_true_all[fold_data['val_idx']] = fold_data['y_val']
                
            # Individual Model Optimization (just for reporting)
            best_f1, _, _ = optimize_thresholds(pd.Series(y_true_all), oof_probas_dict[model_name])
            print(f"[{model_name}] Tuned CV Macro-F1: {best_f1:.4f}")
            mlflow.log_metric(f"{model_name}_cv_macro_f1", best_f1)
            
        print("\n--- Constructing Tuned Ensemble ---")
        
        weights = {
            "LightGBM_Tuned": 1.0,
            "XGBoost_Tuned": 1.0,
            "CatBoost_Tuned": 1.0
        }
        
        ensemble_proba = np.zeros((len_df, 3))
        for name, proba in oof_probas_dict.items():
            ensemble_proba += proba * weights[name]
        ensemble_proba /= sum(weights.values())
        
        print("Optimizing thresholds for the Ensemble...")
        ensemble_f1, best_thresholds, _ = optimize_thresholds(pd.Series(y_true_all), ensemble_proba)
        
        # Calculate baseline unoptimized for comparison
        y_pred_argmax = np.argmax(ensemble_proba, axis=1)
        ensemble_argmax_f1 = f1_score(y_true_all, y_pred_argmax, average='macro')
        
        print(f"Tuned Ensemble (Argmax) Macro-F1    : {ensemble_argmax_f1:.4f}")
        print(f"Tuned Ensemble (Optimized) Macro-F1 : {ensemble_f1:.4f}")
        print(f"Best Multipliers                    : {best_thresholds}")
        
        mlflow.log_metric("ensemble_argmax_macro_f1", ensemble_argmax_f1)
        mlflow.log_metric("ensemble_optimized_macro_f1", ensemble_f1)
        mlflow.log_dict({'threshold_multipliers': best_thresholds.tolist()}, "best_thresholds.json")
        mlflow.log_dict(weights, "ensemble_weights.json")
        
        print("\n=======================================================")
        print(f"🚀 FINAL PHASE 13 TUNED ENSEMBLE CV SCORE: {ensemble_f1:.4f}")
        print("=======================================================")

if __name__ == '__main__':
    run_tuned_ensemble()
