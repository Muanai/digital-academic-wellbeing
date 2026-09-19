import sys
import pandas as pd
import numpy as np
import mlflow

sys.path.insert(0, '.')
from src.preprocessing import apply_canonicalization
from src.features import generate_features
from src.validation import get_cv_strategy, evaluate_model, optimize_thresholds
from src.models import get_baseline_models, build_baseline_pipeline, compute_xgb_sample_weights
from src.utils import seed_everything

def run_experiment():
    seed_everything(42)
    print("Loading data...")
    df_raw = pd.read_csv('data/raw/train.csv')
    
    print("Applying canonicalization and generating features...")
    df = apply_canonicalization(df_raw)
    df_feat = generate_features(df)
    
    target_col = 'digital_academic_wellbeing_level'
    cols_to_drop = ['respondent_id', target_col]
    raw_to_drop = [c.replace('_clean', '') for c in df_feat.columns if c.endswith('_clean')]
    cols_to_drop.extend(raw_to_drop)
    
    y = df_feat[target_col].map({'Low': 0, 'Medium': 1, 'High': 2})
    X = df_feat.drop(columns=cols_to_drop, errors='ignore')
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    cv = get_cv_strategy(n_splits=5, random_state=42)
    models = get_baseline_models(random_state=42)
    xgb_sample_weight = compute_xgb_sample_weights(y)
    
    oof_probas_dict = {name: np.zeros((len(df), 3)) for name in models.keys()}
    
    mlflow.set_experiment("Phase9_Grand_Fusion")
    with mlflow.start_run(run_name="Exp_G_GroupAgg_MultiSeed"):
        mlflow.log_param("experiment", "Exp_G_Phase9")
        
        results = []
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            pipeline = build_baseline_pipeline(model, numeric_features, categorical_features)
            
            sw = xgb_sample_weight if "XGBoost" in model_name else None
            overall_score, oof_preds, oof_proba = evaluate_model(pipeline, X, y, cv, sample_weight=sw)
            
            oof_probas_dict[model_name] = oof_proba
            
            best_f1, best_weights, best_preds = optimize_thresholds(y, oof_proba)
            final_score = best_f1 if best_f1 is not None else overall_score
            
            results.append({'Model': model_name, 'Macro-F1 (Standalone)': final_score})
            mlflow.log_metric(f"{model_name}_macro_f1", final_score)
            print(f"{model_name} Score: {final_score:.4f}")
            
        res_df = pd.DataFrame(results).sort_values('Macro-F1 (Standalone)', ascending=False)
        print("\n=== EXP G: STANDALONE MODEL RESULTS ===")
        print(res_df.to_string(index=False))
        
        # Soft-Voting Ensemble (Average across all models)
        print("\nEvaluating Soft Voting Ensemble (Simple Averaging all 10+ models)...")
        blended_proba_simple = np.zeros((len(y), 3))
        for name in models.keys():
            blended_proba_simple += oof_probas_dict[name]
        blended_proba_simple /= len(models.keys())
        
        from sklearn.metrics import f1_score
        ensemble_f1_default = f1_score(y, np.argmax(blended_proba_simple, axis=1), average='macro')
        print(f"\nSimple Ensemble Macro-F1 (Default Thresholds): {ensemble_f1_default:.4f}")

        # Optimize thresholds on the simple ensemble probabilities
        best_ens_f1_simple, best_ens_multipliers_simple, _ = optimize_thresholds(y, blended_proba_simple)
        print(f"Simple Ensemble Macro-F1 (Optimized Thresholds): {best_ens_f1_simple:.4f}")
        
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
            
            blended_proba = np.zeros((len(y), 3))
            for name in models.keys():
                blended_proba += oof_probas_dict[name] * (weights[name] / total_weight)
                
            # Optimize thresholds for this weighted probability
            # We use a fast approximate check here or just argmax if it's too slow.
            # For exactness, we optimize thresholds on the weighted proba
            best_f1, _, _ = optimize_thresholds(y, blended_proba)
            if best_f1 is None:
                return f1_score(y, np.argmax(blended_proba, axis=1), average='macro')
            return best_f1
            
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=150)
        
        print(f"\nBest Optuna Weighted Ensemble Macro-F1: {study.best_value:.4f}")
        print("Best Weights:")
        best_weights_raw = study.best_params
        total_best_weight = sum(best_weights_raw.values())
        final_weights = {name: best_weights_raw[f'weight_{name}'] / total_best_weight for name in models.keys()}
        for k, v in final_weights.items():
            print(f"  {k}: {v:.4f}")
            
        mlflow.log_metric("Ensemble_Macro_F1_Optimized", study.best_value)

if __name__ == '__main__':
    run_experiment()
