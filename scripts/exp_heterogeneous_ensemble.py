import sys
import pandas as pd
import numpy as np
import mlflow

sys.path.insert(0, '.')
from src.preprocessing import apply_canonicalization
from src.features import generate_features, get_model_specific_features
from src.validation import get_cv_strategy, evaluate_model, optimize_thresholds
from src.models import get_baseline_models, build_baseline_pipeline, compute_xgb_sample_weights
from src.utils import seed_everything

def run_experiment():
    seed_everything(42)
    print("Loading data...")
    df_raw = pd.read_csv('data/raw/train.csv')
    
    print("Applying base preprocessing...")
    df = apply_canonicalization(df_raw)
    
    # Target and drops
    target_col = 'digital_academic_wellbeing_level'
    cols_to_drop = ['respondent_id', target_col]
    raw_to_drop = [c.replace('_clean', '') for c in df.columns if c.endswith('_clean')]
    cols_to_drop.extend(raw_to_drop)
    y = df[target_col].map({'Low': 0, 'Medium': 1, 'High': 2})
    
    cv = get_cv_strategy(n_splits=5, random_state=42)
    models = get_baseline_models(random_state=42)
    xgb_sample_weight = compute_xgb_sample_weights(y)
    
    # We will collect OOF probabilities for the ensemble
    oof_probas_dict = {name: np.zeros((len(df), 3)) for name in models.keys()}
    
    mlflow.set_experiment("Phase8_Advanced")
    with mlflow.start_run(run_name="Exp_F_Heterogeneous_Ensemble"):
        mlflow.log_param("experiment", "Exp_F_Heterogeneous")
        
        results = []
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            model_type = "tree" if model_name in ["CatBoost", "XGBoost", "LightGBM", "RandomForest", "ExtraTrees"] else "linear"
            
            # Generate features SPECIFIC to this model type
            df_feat = get_model_specific_features(df, model_type=model_type)
            
            # Prepare X specific to this model
            X = df_feat.drop(columns=cols_to_drop, errors='ignore')
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            pipeline = build_baseline_pipeline(model, numeric_features, categorical_features)
            
            sw = xgb_sample_weight if model_name == "XGBoost" else None
            overall_score, oof_preds, oof_proba = evaluate_model(pipeline, X, y, cv, sample_weight=sw)
            
            # Store for ensemble
            oof_probas_dict[model_name] = oof_proba
            
            best_f1, best_weights, best_preds = optimize_thresholds(y, oof_proba)
            final_score = best_f1 if best_f1 is not None else overall_score
            
            results.append({'Model': model_name, 'Macro-F1 (Standalone)': final_score})
            mlflow.log_metric(f"{model_name}_macro_f1", final_score)
            print(f"{model_name} Score: {final_score:.4f} (Type: {model_type})")
            
        res_df = pd.DataFrame(results).sort_values('Macro-F1 (Standalone)', ascending=False)
        print("\n=== EXP F: STANDALONE MODEL RESULTS ===")
        print(res_df.to_string(index=False))
        
        # Now, create Soft-Voting Ensemble using Optuna weighting!
        print("\nOptimizing Soft Voting Ensemble Weights...")
        import optuna
        from sklearn.metrics import f1_score
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            weights = []
            for name in models.keys():
                weights.append(trial.suggest_float(f"weight_{name}", 0, 1))
            
            # Normalize weights
            weights = np.array(weights)
            weights /= weights.sum()
            
            # Blend OOF probabilities
            blended_proba = np.zeros((len(y), 3))
            for i, name in enumerate(models.keys()):
                blended_proba += oof_probas_dict[name] * weights[i]
                
            # Default argmax evaluation for weighting
            preds = np.argmax(blended_proba, axis=1)
            return f1_score(y, preds, average='macro')

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=200, show_progress_bar=False)

        best_weights = study.best_params
        total = sum(best_weights.values())
        norm_weights = {k: v / total for k, v in best_weights.items()}

        print("\nOptimal Ensemble Weights:")
        for k, v in norm_weights.items():
            print(f"  {k.replace('weight_', '')}: {v:.4f}")
            mlflow.log_param(f"ensemble_{k}", v)

        # Evaluate ensemble with default argmax
        blended_proba = np.zeros((len(y), 3))
        for name in models.keys():
            blended_proba += oof_probas_dict[name] * norm_weights[f"weight_{name}"]
            
        ensemble_f1_default = f1_score(y, np.argmax(blended_proba, axis=1), average='macro')
        print(f"\nEnsemble Macro-F1 (Default Thresholds): {ensemble_f1_default:.4f}")

        # Optimize thresholds on the ensemble probabilities using Nelder-Mead
        best_ens_f1, best_ens_multipliers, _ = optimize_thresholds(y, blended_proba)
        
        mlflow.log_metric("Ensemble_Macro_F1_Optimized", best_ens_f1)
        print(f"Ensemble Macro-F1 (Optimized Thresholds): {best_ens_f1:.4f}")
        print(f"Optimal Threshold Multipliers: {best_ens_multipliers}")

if __name__ == '__main__':
    run_experiment()
