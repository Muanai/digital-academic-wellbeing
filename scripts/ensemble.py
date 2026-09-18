import os
import sys
import numpy as np
import pandas as pd
import optuna
import mlflow
import json
from pathlib import Path
from sklearn.metrics import classification_report, f1_score

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.validation import optimize_thresholds

def load_data(artifact_dir):
    """Loads true labels and OOF probabilities."""
    error_df_path = os.path.join(artifact_dir, "error_df.csv")
    if not os.path.exists(error_df_path):
        raise FileNotFoundError(f"Missing {error_df_path}. Run error_analysis.py first.")
        
    error_df = pd.read_csv(error_df_path)
    y_true = error_df['y_true'].values
    
    oof_probas = {}
    for file in os.listdir(artifact_dir):
        if file.endswith("_oof_proba.npy"):
            model_name = file.replace("_oof_proba.npy", "")
            oof_probas[model_name] = np.load(os.path.join(artifact_dir, file))
            
    return y_true, oof_probas

def objective(trial, y_true, models, oof_probas):
    """Optuna objective for finding best ensemble weights."""
    weights = []
    for model in models:
        weights.append(trial.suggest_float(model, 0.0, 1.0))
        
    # Normalize weights so they sum to 1
    weight_sum = sum(weights)
    if weight_sum == 0:
        return 0.0 # Invalid
    weights = [w / weight_sum for w in weights]
    
    # Calculate weighted probabilities
    ensemble_proba = np.zeros_like(oof_probas[models[0]])
    for w, model in zip(weights, models):
        ensemble_proba += w * oof_probas[model]
        
    # Optimize thresholds
    best_f1, _, _ = optimize_thresholds(y_true, ensemble_proba)
    
    return best_f1

def run_ensemble_scenario(scenario_name, models, y_true, oof_probas, n_trials=100):
    """Runs Optuna for a specific ensemble scenario and logs to MLflow."""
    print(f"\n{'='*50}\nRunning {scenario_name}\nModels: {', '.join(models)}\n{'='*50}")
    
    # Check if all models are available
    missing = [m for m in models if m not in oof_probas]
    if missing:
        print(f"Skipping {scenario_name}. Missing OOF probas for: {missing}")
        return None
        
    # Setup Optuna study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, y_true, models, oof_probas), n_trials=n_trials)
    
    best_weights_raw = study.best_params
    # Normalize best weights
    weight_sum = sum(best_weights_raw.values())
    best_weights = {k: v / weight_sum for k, v in best_weights_raw.items()}
    
    # Compute final ensemble proba with best weights
    final_proba = np.zeros_like(oof_probas[models[0]])
    for model in models:
        final_proba += best_weights[model] * oof_probas[model]
        
    # Default argmax evaluation (Raw F1)
    raw_preds = np.argmax(final_proba, axis=1)
    raw_f1 = f1_score(y_true, raw_preds, average='macro')
    
    # Optimized Threshold evaluation
    opt_f1, opt_multipliers, opt_preds = optimize_thresholds(y_true, final_proba)
    
    # Log to MLflow
    with mlflow.start_run(run_name=scenario_name):
        # Parameters
        mlflow.log_param("scenario", scenario_name)
        mlflow.log_param("models", ",".join(models))
        for k, v in best_weights.items():
            mlflow.log_param(f"weight_{k}", v)
            
        # Metrics
        mlflow.log_metric("oof_macro_f1_raw", raw_f1)
        mlflow.log_metric("oof_macro_f1_optimized", opt_f1)
        
        # Calculate per-class F1 for detailed insight
        report = classification_report(y_true, opt_preds, target_names=['Low', 'Medium', 'High'], output_dict=True)
        mlflow.log_metric("f1_low", report['Low']['f1-score'])
        mlflow.log_metric("f1_medium", report['Medium']['f1-score'])
        mlflow.log_metric("f1_high", report['High']['f1-score'])
        
        # Save multipliers as json artifact
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump({'multipliers': list(opt_multipliers), 'weights': best_weights}, f, indent=4)
            temp_path = f.name
        mlflow.log_artifact(temp_path, "ensemble_config")
        os.remove(temp_path)
    
    print(f"Best Weights: {best_weights}")
    print(f"Raw Macro-F1: {raw_f1:.4f}")
    print(f"Optimized Macro-F1: {opt_f1:.4f}")
    print(f"Threshold Multipliers: {opt_multipliers}")
    print("\nClassification Report (Optimized):")
    print(classification_report(y_true, opt_preds, target_names=['Low', 'Medium', 'High']))
    
    return {
        "scenario": scenario_name,
        "f1": opt_f1,
        "weights": best_weights,
        "multipliers": opt_multipliers
    }

def main():
    artifact_dir = "experiments/phase5_artifacts"
    y_true, oof_probas = load_data(artifact_dir)
    
    # Set MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = "Phase_6_Ensemble"
    
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    
    scenarios = {
        "Scenario_1_Full": ["CatBoost", "ExtraTrees", "LightGBM", "LogisticRegression", "MLPClassifier", "RandomForest", "XGBoost"],
        "Scenario_2_Golden": ["CatBoost", "XGBoost", "LogisticRegression", "MLPClassifier"],
        "Scenario_3_Top3": ["CatBoost", "XGBoost", "LightGBM"],
        "Scenario_4_TreeBased": ["CatBoost", "XGBoost", "LightGBM", "RandomForest", "ExtraTrees"],
        "Scenario_5_SimpleMix": ["CatBoost", "LogisticRegression"]
    }
    
    results = []
    for name, models in scenarios.items():
        # Using 200 trials for rigorous weight optimization
        res = run_ensemble_scenario(name, models, y_true, oof_probas, n_trials=200)
        if res:
            results.append(res)
            
    if results:
        results.sort(key=lambda x: x['f1'], reverse=True)
        best = results[0]
        print(f"\n{'='*50}\nOVERALL BEST ENSEMBLE: {best['scenario']}\nMacro-F1: {best['f1']:.4f}\n{'='*50}")

if __name__ == "__main__":
    main()
