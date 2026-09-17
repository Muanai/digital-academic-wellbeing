import os
import sys
import pandas as pd
import numpy as np
import mlflow
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import apply_canonicalization
from src.features import generate_features
from src.validation import get_cv_strategy, evaluate_model, generate_error_analysis, optimize_thresholds
from src.models import get_baseline_models, build_baseline_pipeline, compute_xgb_sample_weights

def main():
    print("Starting Phase 3 Feature Engineering Evaluation...")
    data_dir = Path("data/raw")
    train_path = data_dir / "train.csv"
    
    if not train_path.exists():
        print(f"Error: Data not found at {train_path}")
        return
        
    df_train_raw = pd.read_csv(train_path)
    
    # 1. Apply Canonicalization (Phase 1)
    df_train_clean = apply_canonicalization(df_train_raw)
    
    # 2. Apply Feature Engineering (Phase 3)
    df_train_engineered = generate_features(df_train_clean)
    
    # Target definition
    target_col = 'digital_academic_wellbeing_level'
    
    # Map target to integers (XGBoost requires 0, 1, 2)
    target_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df_train_engineered[target_col] = df_train_engineered[target_col].map(target_map)
    
    # We will use the _clean and engineered columns for model training, and drop raw columns with clean equivalents
    cols_to_drop = ['respondent_id', target_col]
    raw_to_drop = [c.replace('_clean', '') for c in df_train_engineered.columns if c.endswith('_clean')]
    cols_to_drop.extend(raw_to_drop)
    
    X = df_train_engineered.drop(columns=cols_to_drop, errors='ignore')
    y = df_train_engineered[target_col]
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 2. Setup CV
    cv = get_cv_strategy(n_splits=5, random_state=42)
    
    # 3. Get Baseline Models
    models = get_baseline_models(random_state=42)
    
    # 4. Setup MLflow
    mlflow.set_experiment("Phase_3_Feature_Engineering")
    
    results = []
    oof_predictions_df = pd.DataFrame(index=df_train_raw.index)
    oof_probas = {}  # model_name -> np.ndarray [n_samples, n_classes]
    
    # Directory to store local artifacts before logging to MLflow
    artifact_dir = "experiments/artifacts"
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Precompute sample_weight for XGBoost (class balancing workaround)
    xgb_sample_weight = compute_xgb_sample_weights(y)
    
    with mlflow.start_run(run_name="Phase_3_6_Architecture_Upgrade"):
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("cv_seed", 42)
        mlflow.log_param("numeric_features_count", len(numeric_features))
        mlflow.log_param("categorical_features_count", len(categorical_features))
        
        for model_name, model in models.items():
            print(f"\n--- Evaluating {model_name} ---")
            
            with mlflow.start_run(run_name=f"Model_{model_name}", nested=True):
                pipeline = build_baseline_pipeline(model, numeric_features, categorical_features)
                
                # XGBoost gets sample_weight for class balancing
                sw = xgb_sample_weight if model_name == "XGBoost" else None
                overall_score, oof_preds, oof_proba = evaluate_model(pipeline, X, y, cv, sample_weight=sw)
                
                # Save OOF predictions for ensemble later
                oof_predictions_df[f"{model_name}_pred"] = oof_preds
                if oof_proba is not None:
                    oof_probas[model_name] = oof_proba
                    
                # Threshold Optimization
                best_f1, best_weights, best_preds = optimize_thresholds(y, oof_proba)
                if best_f1 is not None:
                    print(f"  Optimized OOF Macro-F1: {best_f1:.4f} (Weights: {best_weights})")
                else:
                    best_f1 = overall_score
                
                mlflow.log_param("model_name", model_name)
                mlflow.log_metric("macro_f1_default", overall_score)
                mlflow.log_metric("macro_f1_tuned", best_f1)
                
                results.append({
                    'Model': model_name, 
                    'Macro-F1 (Default)': overall_score,
                    'Macro-F1 (Tuned)': best_f1
                })
                
                # Error Analysis for EVERY model (User's request)
                cm_path, misclassified_mask = generate_error_analysis(
                    y_true=y, 
                    y_pred=oof_preds, 
                    model_name=model_name, 
                    output_dir=artifact_dir
                )
                
                # Save Misclassifications to CSV
                misclassified_df = df_train_raw[misclassified_mask].copy()
                misclassified_df['True_Label'] = y[misclassified_mask]
                misclassified_df['Predicted_Label'] = oof_preds[misclassified_mask]
                
                csv_path = os.path.join(artifact_dir, f"{model_name}_misclassified.csv")
                misclassified_df.to_csv(csv_path, index=False)
                
                # Log to MLflow
                mlflow.log_artifact(cm_path)
                mlflow.log_artifact(csv_path)

        # 5. Summarize Results
        df_results = pd.DataFrame(results).sort_values(by='Macro-F1 (Tuned)', ascending=False)
        print("\n=== Phase 3.6 Architecture Upgrade Results ===")
        print(df_results)
        
        # Save OOF predictions DataFrame
        oof_path = os.path.join(artifact_dir, "oof_predictions.csv")
        oof_predictions_df['True_Label'] = y
        oof_predictions_df.to_csv(oof_path, index=False)
        mlflow.log_artifact(oof_path)
        
        # Save OOF probabilities per model
        for model_name, proba in oof_probas.items():
            proba_path = os.path.join(artifact_dir, f"{model_name}_oof_proba.npy")
            np.save(proba_path, proba)
            mlflow.log_artifact(proba_path)
        
        # Save Report locally
        report_path = r"C:\Users\user\.gemini\antigravity-ide\brain\61d1d5ec-f855-42cb-8916-4ccc318a689b\phase3_feature_engineering_report.md"
        
        md_table = "| Model | Macro-F1 (Default) | Macro-F1 (Tuned) |\n|---|---|---|\n"
        for _, row in df_results.iterrows():
            md_table += f"| {row['Model']} | {row['Macro-F1 (Default)']:.4f} | {row['Macro-F1 (Tuned)']:.4f} |\n"
            
        report_lines = [
            "# Phase 3.6 Architecture Upgrade MLflow Results",
            "\n",
            md_table,
            "\n",
            "## Artifacts Generated",
            f"- Confusion Matrices saved to `{artifact_dir}` and logged to MLflow.",
            f"- Misclassification CSVs saved to `{artifact_dir}` and logged to MLflow.",
            f"- All OOF Predictions saved to `{oof_path}` for future ensembling."
        ]
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
            
        print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    main()
