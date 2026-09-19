import sys
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import f1_score

sys.path.insert(0, '.')
from src.preprocessing import apply_canonicalization
from src.features import generate_features
from src.validation import get_cv_strategy, optimize_thresholds
from src.models import get_baseline_models, build_baseline_pipeline, compute_xgb_sample_weights
from src.utils import seed_everything

def run_experiment():
    seed_everything(42)
    print("Loading data...")
    df_raw = pd.read_csv('data/raw/train.csv')
    
    cv = get_cv_strategy(n_splits=5, random_state=42)
    models = get_baseline_models(random_state=42)
    
    oof_probas_dict = {name: np.zeros((len(df_raw), 3)) for name in models.keys()}
    
    target_col = 'digital_academic_wellbeing_level'
    label_map = {'Low': 0, 'Medium': 1, 'High': 2}
    
    mlflow.set_experiment("Phase10_Leakage_Free")
    with mlflow.start_run(run_name="Exp_H_StrictCV_WinningDomain"):
        mlflow.log_param("experiment", "Exp_H_Phase10")
        
        results = []
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            
            oof_proba = np.zeros((len(df_raw), 3))
            oof_preds = np.zeros(len(df_raw))
            y_true_all = np.zeros(len(df_raw))
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(df_raw, df_raw[target_col])):
                df_train = df_raw.iloc[train_idx].copy()
                df_val = df_raw.iloc[val_idx].copy()
                
                # Preprocessing inside the fold
                df_train = apply_canonicalization(df_train)
                df_val = apply_canonicalization(df_val)
                
                # Feature engineering inside the fold, preventing leakage!
                # df_train acts as the stats_df for df_val
                df_feat_train = generate_features(df_train)
                df_feat_val = generate_features(df_val, stats_df=df_feat_train)
                
                y_train = df_feat_train[target_col].map(label_map)
                y_val = df_feat_val[target_col].map(label_map)
                
                cols_to_drop = ['respondent_id', target_col]
                raw_to_drop = [c.replace('_clean', '') for c in df_feat_train.columns if c.endswith('_clean')]
                cols_to_drop.extend(raw_to_drop)
                
                X_train = df_feat_train.drop(columns=cols_to_drop, errors='ignore')
                X_val = df_feat_val.drop(columns=cols_to_drop, errors='ignore')
                
                # Align columns just in case
                for c in X_train.columns:
                    if c not in X_val.columns:
                        X_val[c] = 0
                X_val = X_val[X_train.columns]
                
                numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
                
                pipeline = build_baseline_pipeline(model, numeric_features, categorical_features)
                
                if "XGBoost" in model_name:
                    sw_train = compute_xgb_sample_weights(y_train)
                    pipeline.fit(X_train, y_train, classifier__sample_weight=sw_train)
                else:
                    pipeline.fit(X_train, y_train)
                
                proba = pipeline.predict_proba(X_val)
                preds = np.argmax(proba, axis=1)
                
                oof_proba[val_idx] = proba
                oof_preds[val_idx] = preds
                y_true_all[val_idx] = y_val
                
                score = f1_score(y_val, preds, average='macro')
                # print(f"  Fold {fold+1} Macro-F1: {score:.4f}")
            
            oof_probas_dict[model_name] = oof_proba
            
            overall_score = f1_score(y_true_all, oof_preds, average='macro')
            # print(f"  Overall OOF Macro-F1: {overall_score:.4f}")
            
            y_series = pd.Series(y_true_all)
            best_f1, best_weights, best_preds = optimize_thresholds(y_series, oof_proba)
            final_score = best_f1 if best_f1 is not None else overall_score
            
            results.append({'Model': model_name, 'Macro-F1 (Standalone)': final_score})
            mlflow.log_metric(f"{model_name}_macro_f1", final_score)
            print(f"{model_name} Score (Optimized): {final_score:.4f}")
            
        res_df = pd.DataFrame(results).sort_values('Macro-F1 (Standalone)', ascending=False)
        print("\n=== EXP H: STANDALONE MODEL RESULTS ===")
        print(res_df.to_string(index=False))
        
        # Soft-Voting Ensemble (Average across all models)
        print("\nEvaluating Soft Voting Ensemble (Simple Averaging all models)...")
        blended_proba_simple = np.zeros((len(df_raw), 3))
        for name in models.keys():
            blended_proba_simple += oof_probas_dict[name]
        blended_proba_simple /= len(models.keys())
        
        y_series = pd.Series(y_true_all)
        ensemble_f1_default = f1_score(y_series, np.argmax(blended_proba_simple, axis=1), average='macro')
        print(f"\nSimple Ensemble Macro-F1 (Default Thresholds): {ensemble_f1_default:.4f}")

        # Optimize thresholds on the simple ensemble probabilities
        best_ens_f1_simple, best_ens_multipliers_simple, _ = optimize_thresholds(y_series, blended_proba_simple)
        print(f"Simple Ensemble Macro-F1 (Optimized Thresholds): {best_ens_f1_simple:.4f}")
        mlflow.log_metric("Ensemble_Macro_F1_Optimized", best_ens_f1_simple)
        
        print("\nRunning Optuna to find optimal ensemble weights (like Phase 6)...")
        import optuna
        
        def objective(trial):
            # Propose weights for each model
            weights = {}
            for name in models.keys():
                weights[name] = trial.suggest_float(f'weight_{name}', 0, 1)
                
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight == 0:
                return 0.0
            
            blended_proba = np.zeros((len(y_true_all), 3))
            for name in models.keys():
                blended_proba += oof_probas_dict[name] * (weights[name] / total_weight)
                
            best_f1, _, _ = optimize_thresholds(y_series, blended_proba)
            if best_f1 is None:
                return f1_score(y_true_all, np.argmax(blended_proba, axis=1), average='macro')
            return best_f1
            
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=100)
        
        print(f"\nBest Optuna Weighted Ensemble Macro-F1: {study.best_value:.4f}")
        print("Best Weights:")
        best_weights_raw = study.best_params
        total_best_weight = sum(best_weights_raw.values())
        final_weights = {name: best_weights_raw[f'weight_{name}'] / total_best_weight for name in models.keys()}
        for k, v in final_weights.items():
            print(f"  {k}: {v:.4f}")
            
        mlflow.log_metric("Ensemble_Macro_F1_OptunaWeighted", study.best_value)

if __name__ == '__main__':
    run_experiment()
