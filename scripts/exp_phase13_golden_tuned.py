import sys
import json
import warnings
import pandas as pd
import numpy as np
import optuna
import mlflow
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.preprocessing import apply_canonicalization
from src.features import get_model_specific_features
from src.validation import get_cv_strategy, evaluate_model, optimize_thresholds
from src.models import get_baseline_models, build_baseline_pipeline
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

def run_golden_tuned_ensemble():
    seed_everything(42)
    print("Loading data and tuned hyperparameters...")
    df_raw = pd.read_csv('data/raw/train.csv')
    best_params = load_best_params()
    
    print("Applying base preprocessing...")
    df = apply_canonicalization(df_raw)
    
    target_col = 'digital_academic_wellbeing_level'
    cols_to_drop = ['respondent_id', target_col]
    raw_to_drop = [c.replace('_clean', '') for c in df.columns if c.endswith('_clean')]
    cols_to_drop.extend(raw_to_drop)
    y = df[target_col].map({'Low': 0, 'Medium': 1, 'High': 2})
    len_df = len(df_raw)
    y_true_all = y.values
    
    cv = get_cv_strategy(n_splits=5, random_state=42)
    
    # 1. Load Phase 6 Golden Models
    golden_models = get_baseline_models(random_state=42)
    
    # 2. Add Phase 12 Tuned Models
    tuned_models = {
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
    
    oof_probas_dict = {}
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Phase13_Golden_Tuned_Ensemble")
    
    with mlflow.start_run(run_name="Golden_Tuned_Grand_Ensemble"):
        # ==========================================
        # STEP A: Evaluate Phase 6 Golden Models (Ablated Features)
        # ==========================================
        print("\n--- Evaluating Phase 6 Golden Models (Ablated Features) ---")
        for model_name, model in golden_models.items():
            print(f"  > {model_name}...")
            model_type = "tree" if any(t in model_name for t in ["CatBoost", "XGBoost", "LightGBM", "RandomForest", "ExtraTrees"]) else "linear"
            
            df_feat = get_model_specific_features(df, model_type=model_type)
            X = df_feat.drop(columns=cols_to_drop, errors='ignore')
            numeric_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
            categorical_features = [c for c in X.columns if c not in numeric_features]
            
            pipeline = build_baseline_pipeline(model, numeric_features, categorical_features)
            
            if "XGBoost" in model_name:
                from src.models import compute_xgb_sample_weights
                sw = compute_xgb_sample_weights(y)
                _, _, oof_proba = evaluate_model(pipeline, X, y, cv, sample_weight=sw)
            else:
                _, _, oof_proba = evaluate_model(pipeline, X, y, cv)
                
            oof_probas_dict[model_name] = oof_proba
            
        # ==========================================
        # STEP B: Evaluate Phase 12 Tuned Models (Full Features)
        # ==========================================
        print("\n--- Evaluating Phase 12 Tuned Models (Full Features) ---")
        label_map = {'Low': 0, 'Medium': 1, 'High': 2}
        folds_data = precompute_folds(df_raw, target_col, label_map, cv)
        
        for model_name, model in tuned_models.items():
            print(f"  > {model_name}...")
            oof_proba = np.zeros((len_df, 3))
            
            for fold_data in folds_data:
                if "XGBoost" in model_name:
                    model.fit(fold_data['X_train'], fold_data['y_train'], sample_weight=fold_data['sw_train'])
                else:
                    model.fit(fold_data['X_train'], fold_data['y_train'])
                    
                proba = model.predict_proba(fold_data['X_val'])
                oof_proba[fold_data['val_idx']] = proba
                
            oof_probas_dict[model_name] = oof_proba
            
        # ==========================================
        # STEP C: Optuna Ensemble Weight Optimization
        # ==========================================
        print("\n--- Optimizing Golden + Tuned Soft-Voting Weights via Optuna ---")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            weights = []
            model_keys = list(oof_probas_dict.keys())
            for name in model_keys:
                weights.append(trial.suggest_float(f"weight_{name}", 0, 1))
            
            weights = np.array(weights)
            weights /= weights.sum()
            
            blended_proba = np.zeros((len_df, 3))
            for i, name in enumerate(model_keys):
                blended_proba += oof_probas_dict[name] * weights[i]
                
            # Optuna uses unoptimized argmax F1 to find best probability blend distribution
            preds = np.argmax(blended_proba, axis=1)
            return f1_score(y_true_all, preds, average='macro')

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=300)
        
        best_weights = study.best_params
        total = sum(best_weights.values())
        norm_weights = {k.replace('weight_', ''): v / total for k, v in best_weights.items()}
        
        print("\nBest Optuna Model Weights:")
        for k, v in sorted(norm_weights.items(), key=lambda item: item[1], reverse=True):
            print(f"  {k}: {v:.4f}")
            
        # ==========================================
        # STEP D: Final Ensemble & Nelder-Mead Thresholding
        # ==========================================
        ensemble_proba = np.zeros((len_df, 3))
        for name, proba in oof_probas_dict.items():
            ensemble_proba += proba * norm_weights[name]
            
        print("\nApplying Nelder-Mead Threshold Optimization to the Final Blend...")
        ensemble_f1, best_thresholds, _ = optimize_thresholds(pd.Series(y_true_all), ensemble_proba)
        
        y_pred_argmax = np.argmax(ensemble_proba, axis=1)
        ensemble_argmax_f1 = f1_score(y_true_all, y_pred_argmax, average='macro')
        
        print(f"Golden+Tuned Ensemble (Argmax) Macro-F1    : {ensemble_argmax_f1:.4f}")
        print(f"Golden+Tuned Ensemble (Optimized) Macro-F1 : {ensemble_f1:.4f}")
        print(f"Best Multipliers                           : {best_thresholds}")
        
        mlflow.log_metric("ensemble_argmax_macro_f1", ensemble_argmax_f1)
        mlflow.log_metric("ensemble_optimized_macro_f1", ensemble_f1)
        mlflow.log_dict({'threshold_multipliers': best_thresholds.tolist()}, "best_thresholds.json")
        mlflow.log_dict(norm_weights, "ensemble_weights.json")
        
        print("\n=======================================================")
        print(f"🏆 FINAL PHASE 13 GOLDEN+TUNED CV SCORE: {ensemble_f1:.4f}")
        print("=======================================================")

if __name__ == '__main__':
    run_golden_tuned_ensemble()
