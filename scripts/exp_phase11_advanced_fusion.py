import sys
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import f1_score

sys.path.insert(0, '.')
from src.preprocessing import apply_canonicalization
from src.features import generate_features
from src.validation import get_cv_strategy, optimize_regression_thresholds, optimize_thresholds
from src.models import get_regression_models, build_baseline_pipeline, compute_xgb_sample_weights
from src.dl_models import TabularResNetWrapper
from src.utils import seed_everything

def run_experiment():
    seed_everything(42)
    print("Loading data...")
    df_raw = pd.read_csv('data/raw/train.csv')
    
    cv = get_cv_strategy(n_splits=5, random_state=42)
    
    # We will test Regression models and TabularResNet
    models = get_regression_models(random_state=42)
    
    # Add TabularResNet (we'll set features later dynamically)
    models["TabularResNet"] = TabularResNetWrapper(numeric_features=[], categorical_features=[])
    
    target_col = 'digital_academic_wellbeing_level'
    label_map = {'Low': 0, 'Medium': 1, 'High': 2}
    
    mlflow.set_experiment("Phase11_Advanced_Fusion")
    with mlflow.start_run(run_name="Exp_I_Regression_and_DL"):
        mlflow.log_param("experiment", "Exp_I_Phase11")
        
        results = []
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            
            # For TabularResNet we store probabilities, for Regression we store continuous predictions
            is_regressor = "Reg" in model_name
            is_dl = "TabularResNet" in model_name
            
            if is_regressor:
                oof_preds_cont = np.zeros(len(df_raw))
            elif is_dl:
                oof_proba = np.zeros((len(df_raw), 3))
                
            oof_preds = np.zeros(len(df_raw))
            y_true_all = np.zeros(len(df_raw))
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(df_raw, df_raw[target_col])):
                df_train = df_raw.iloc[train_idx].copy()
                df_val = df_raw.iloc[val_idx].copy()
                
                df_train = apply_canonicalization(df_train)
                df_val = apply_canonicalization(df_val)
                
                df_feat_train = generate_features(df_train)
                df_feat_val = generate_features(df_val, stats_df=df_feat_train)
                
                y_train = df_feat_train[target_col].map(label_map)
                y_val = df_feat_val[target_col].map(label_map)
                
                cols_to_drop = ['respondent_id', target_col]
                raw_to_drop = [c.replace('_clean', '') for c in df_feat_train.columns if c.endswith('_clean')]
                cols_to_drop.extend(raw_to_drop)
                
                X_train = df_feat_train.drop(columns=cols_to_drop, errors='ignore')
                X_val = df_feat_val.drop(columns=cols_to_drop, errors='ignore')
                
                for c in X_train.columns:
                    if c not in X_val.columns:
                        X_val[c] = 0
                X_val = X_val[X_train.columns]
                
                numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
                
                if is_dl:
                    model.numeric_features = numeric_features
                    model.categorical_features = categorical_features
                    pipeline = model
                else:
                    pipeline = build_baseline_pipeline(model, numeric_features, categorical_features)
                
                if "XGBoost" in model_name:
                    # For regressor, we don't necessarily use class weights, but we can pass sample weights
                    sw_train = compute_xgb_sample_weights(y_train)
                    pipeline.fit(X_train, y_train, classifier__sample_weight=sw_train)
                else:
                    pipeline.fit(X_train, y_train)
                
                if is_regressor:
                    preds_cont = pipeline.predict(X_val)
                    oof_preds_cont[val_idx] = preds_cont
                elif is_dl:
                    proba = pipeline.predict_proba(X_val)
                    oof_proba[val_idx] = proba
                    
                y_true_all[val_idx] = y_val
                print(f"  Fold {fold+1} Finished")
            
            y_series = pd.Series(y_true_all)
            
            if is_regressor:
                best_f1, best_thresholds, final_preds = optimize_regression_thresholds(y_series, oof_preds_cont)
                print(f"  Optimal Thresholds: {best_thresholds}")
            elif is_dl:
                best_f1, best_weights, final_preds = optimize_thresholds(y_series, oof_proba)
                
            results.append({'Model': model_name, 'Macro-F1 (Tuned)': best_f1})
            mlflow.log_metric(f"{model_name}_macro_f1", best_f1)
            print(f"{model_name} Score (Optimized): {best_f1:.4f}")
            
        res_df = pd.DataFrame(results).sort_values('Macro-F1 (Tuned)', ascending=False)
        print("\n=== EXP I: REGRESSION & DL RESULTS ===")
        print(res_df.to_string(index=False))

if __name__ == '__main__':
    run_experiment()
