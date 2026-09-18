import sys
import pandas as pd
import numpy as np
import mlflow
from sklearn.base import clone

sys.path.insert(0, '.')
from src.preprocessing import apply_canonicalization
from src.features import generate_features
from src.validation import get_cv_strategy, optimize_thresholds
from src.models import get_baseline_models, build_baseline_pipeline, compute_xgb_sample_weights
from src.utils import seed_everything
from sklearn.metrics import f1_score

def run_experiment():
    seed_everything(42)
    print("Loading data...")
    df_raw = pd.read_csv('data/raw/train.csv')
    
    print("Applying base preprocessing and features...")
    df = apply_canonicalization(df_raw)
    df = generate_features(df)
    
    target_col = 'digital_academic_wellbeing_level'
    cols_to_drop = ['respondent_id', target_col]
    raw_to_drop = [c.replace('_clean', '') for c in df.columns if c.endswith('_clean')]
    cols_to_drop.extend(raw_to_drop)
    
    X = df.drop(columns=cols_to_drop, errors='ignore')
    y = df[target_col].map({'Low': 0, 'Medium': 1, 'High': 2})
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    cv = get_cv_strategy(n_splits=5, random_state=42)
    models = get_baseline_models(random_state=42)
    
    y_geq_med = (y >= 1).astype(int)
    y_geq_high = (y >= 2).astype(int)
    sw_med = compute_xgb_sample_weights(y_geq_med)
    sw_high = compute_xgb_sample_weights(y_geq_high)
    
    mlflow.set_experiment("Phase6.5_Breakthrough")
    with mlflow.start_run(run_name="Exp_B_Ordinal_Classification"):
        mlflow.log_param("experiment", "Exp_B")
        
        results = []
        for model_name, base_model in models.items():
            print(f"\nEvaluating Ordinal {model_name}...")
            oof_preds = pd.Series(index=y.index, dtype=int)
            oof_proba = np.full((len(y), 3), np.nan)
            
            X_arr = X.values if isinstance(X, pd.DataFrame) else X
            
            fold_scores = []
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                # We need dataframe for the ColumnTransformer
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train_med, y_train_high = y_geq_med.iloc[train_idx], y_geq_high.iloc[train_idx]
                
                pipeline_A = build_baseline_pipeline(clone(base_model), numeric_features, categorical_features)
                pipeline_B = build_baseline_pipeline(clone(base_model), numeric_features, categorical_features)
                
                if model_name == "XGBoost":
                    pipeline_A.fit(X_train, y_train_med, classifier__sample_weight=sw_med[train_idx])
                    pipeline_B.fit(X_train, y_train_high, classifier__sample_weight=sw_high[train_idx])
                else:
                    pipeline_A.fit(X_train, y_train_med)
                    pipeline_B.fit(X_train, y_train_high)
                    
                # Get P(Y >= Medium) and P(Y >= High)
                p_geq_med = pipeline_A.predict_proba(X_val)[:, 1]
                p_geq_high = pipeline_B.predict_proba(X_val)[:, 1]
                
                # Derive 3-class probabilities
                p_low = 1.0 - p_geq_med
                p_med = np.maximum(0.0, p_geq_med - p_geq_high)
                p_high = p_geq_high
                
                # Normalize just in case
                probas = np.vstack([p_low, p_med, p_high]).T
                probas = probas / probas.sum(axis=1, keepdims=True)
                
                oof_proba[val_idx] = probas
                preds = np.argmax(probas, axis=1)
                oof_preds.iloc[val_idx] = preds
                
                score = f1_score(y.iloc[val_idx], preds, average='macro')
                fold_scores.append(score)
                
            overall_score = f1_score(y, oof_preds, average='macro')
            best_f1, best_weights, best_preds = optimize_thresholds(y, oof_proba)
            final_score = best_f1 if best_f1 is not None else overall_score
            
            results.append({'Model': model_name, 'Macro-F1 (Tuned)': final_score})
            mlflow.log_metric(f"{model_name}_macro_f1", final_score)
            print(f"{model_name} Score: {final_score:.4f}")
            
        res_df = pd.DataFrame(results).sort_values('Macro-F1 (Tuned)', ascending=False)
        print("\n=== EXP B: ORDINAL CLASSIFICATION RESULTS ===")
        print(res_df.to_string(index=False))

if __name__ == '__main__':
    run_experiment()
