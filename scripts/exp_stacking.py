import sys
import os
import pandas as pd
import numpy as np
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import optuna

sys.path.insert(0, '.')
from src.validation import optimize_thresholds
from src.utils import seed_everything

def run_experiment():
    seed_everything(42)
    print("Loading OOF Probabilities from Phase 3.6...")
    artifact_dir = "experiments/artifacts"
    
    # Check if artifacts exist
    if not os.path.exists(artifact_dir):
        print(f"Error: {artifact_dir} not found. Please ensure Phase 3.6 ran successfully.")
        return
        
    models = ['LightGBM', 'LogisticRegression', 'XGBoost', 'CatBoost', 'RandomForest', 'ExtraTrees', 'MLPClassifier']
    
    oof_features = []
    for m in models:
        path = os.path.join(artifact_dir, f"{m}_oof_proba.npy")
        if os.path.exists(path):
            proba = np.load(path)
            oof_features.append(proba)
        else:
            print(f"Warning: {path} not found.")
            
    if len(oof_features) == 0:
        print("No OOF probabilities found. Aborting.")
        return
        
    # Stack probabilities horizontally (N samples x (num_models * 3 classes))
    X_meta = np.hstack(oof_features)
    print(f"Meta-features shape: {X_meta.shape}")
    
    # Load targets
    oof_df = pd.read_csv(os.path.join(artifact_dir, "oof_predictions.csv"))
    y_true = oof_df['True_Label'].values
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    mlflow.set_experiment("Phase6.5_Breakthrough")
    with mlflow.start_run(run_name="Exp_C_Stacking_MetaLearner"):
        mlflow.log_param("experiment", "Exp_C")
        
        # We will use Optuna to find the best C for LogisticRegression meta-learner
        def objective(trial):
            C = trial.suggest_float('C', 1e-4, 10.0, log=True)
            l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
            
            meta_model = LogisticRegression(
                penalty='elasticnet', C=C, l1_ratio=l1_ratio, solver='saga', 
                class_weight='balanced', max_iter=2000, random_state=42
            )
            
            oof_meta_proba = np.full((len(y_true), 3), np.nan)
            
            for train_idx, val_idx in cv.split(X_meta, y_true):
                X_tr, y_tr = X_meta[train_idx], y_true[train_idx]
                X_va = X_meta[val_idx]
                
                meta_model.fit(X_tr, y_tr)
                oof_meta_proba[val_idx] = meta_model.predict_proba(X_va)
                
            # Optimize thresholds on the meta-probabilities
            best_f1, _, _ = optimize_thresholds(y_true, oof_meta_proba)
            return best_f1 if best_f1 is not None else 0.0
            
        print("\nOptimizing Meta-Learner (Logistic Regression)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        print("\nBest Parameters:")
        print(study.best_params)
        
        # Train final meta-learner with best params for evaluation
        best_model = LogisticRegression(
            penalty='elasticnet', solver='saga', class_weight='balanced', max_iter=2000, random_state=42,
            **study.best_params
        )
        oof_meta_proba = np.full((len(y_true), 3), np.nan)
        oof_meta_preds = np.zeros(len(y_true), dtype=int)
        
        for train_idx, val_idx in cv.split(X_meta, y_true):
            X_tr, y_tr = X_meta[train_idx], y_true[train_idx]
            X_va = X_meta[val_idx]
            
            best_model.fit(X_tr, y_tr)
            oof_meta_proba[val_idx] = best_model.predict_proba(X_va)
            oof_meta_preds[val_idx] = best_model.predict(X_va)
            
        default_score = f1_score(y_true, oof_meta_preds, average='macro')
        tuned_score, best_weights, _ = optimize_thresholds(y_true, oof_meta_proba)
        
        print("\n=== EXP C: STACKING META-LEARNER RESULTS ===")
        print(f"Default Macro-F1: {default_score:.4f}")
        print(f"Tuned Macro-F1:   {tuned_score:.4f}")
        print(f"Optimal Weights:  {best_weights}")
        
        mlflow.log_metric("Stacking_macro_f1", tuned_score)

if __name__ == '__main__':
    run_experiment()
