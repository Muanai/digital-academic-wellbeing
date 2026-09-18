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
from sklearn.model_selection import KFold

def run_experiment():
    seed_everything(42)
    print("Loading data...")
    df_raw = pd.read_csv('data/raw/train.csv')
    
    print("Applying base preprocessing and features...")
    df = apply_canonicalization(df_raw)
    df = generate_features(df)
    
    print("Adding Exp D Extended Group Features...")
    
    # 1. Target Encoding for field_of_study using KFold to prevent leakage
    df['target'] = df['digital_academic_wellbeing_level'].map({'Low': 0, 'Medium': 1, 'High': 2})
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    df['field_of_study_te'] = np.nan
    
    for train_idx, val_idx in kf.split(df):
        tr_fold = df.iloc[train_idx]
        te_map = tr_fold.groupby('field_of_study')['target'].mean()
        # Impute missing categories in val with global mean
        df.loc[val_idx, 'field_of_study_te'] = df.iloc[val_idx]['field_of_study'].map(te_map).fillna(tr_fold['target'].mean())
        
    df.drop(columns=['target'], inplace=True)
    
    # 2. Aggressive Z-Scores within Groups (field_of_study x gender)
    # This captures "relative" behavior within a student's peer group
    df['peer_group'] = df['field_of_study'] + "_" + df['gender_clean']
    
    features_to_group = [
        'sleep_hours', 'procrastination_level_1_10', 'pre_exam_anxiety_level_1_10',
        'caffeine_intake_mg_per_day', 'social_media_hrs_per_day'
    ]
    
    for feat in features_to_group:
        group_mean = df.groupby('peer_group')[feat].transform('mean')
        group_std = df.groupby('peer_group')[feat].transform('std')
        # Avoid division by zero
        df[f"{feat}_peer_zscore"] = (df[feat] - group_mean) / (group_std.replace(0, 1) + 1e-6)
        
    df.drop(columns=['peer_group'], inplace=True)
    
    print(f"Total features after Exp D additions: {df.shape[1]}")
    
    # Prepare X, y
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
    xgb_sample_weight = compute_xgb_sample_weights(y)
    
    mlflow.set_experiment("Phase6.5_Breakthrough")
    with mlflow.start_run(run_name="Exp_D_Extended_Groups"):
        mlflow.log_param("experiment", "Exp_D")
        
        results = []
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            pipeline = build_baseline_pipeline(model, numeric_features, categorical_features)
            
            sw = xgb_sample_weight if model_name == "XGBoost" else None
            overall_score, oof_preds, oof_proba = evaluate_model(pipeline, X, y, cv, sample_weight=sw)
            
            best_f1, best_weights, best_preds = optimize_thresholds(y, oof_proba)
            final_score = best_f1 if best_f1 is not None else overall_score
            
            results.append({'Model': model_name, 'Macro-F1 (Tuned)': final_score})
            mlflow.log_metric(f"{model_name}_macro_f1", final_score)
            print(f"{model_name} Score: {final_score:.4f}")
            
        res_df = pd.DataFrame(results).sort_values('Macro-F1 (Tuned)', ascending=False)
        print("\n=== EXP D: EXTENDED GROUPS RESULTS ===")
        print(res_df.to_string(index=False))

if __name__ == '__main__':
    run_experiment()
