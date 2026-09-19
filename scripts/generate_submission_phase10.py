import sys
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, '.')
from src.preprocessing import apply_canonicalization
from src.features import generate_features
from src.validation import get_cv_strategy, optimize_thresholds
from src.models import get_baseline_models, build_baseline_pipeline, compute_xgb_sample_weights
from src.utils import seed_everything

def run():
    seed_everything(42)
    print("Loading data...")
    df_raw = pd.read_csv('data/raw/train.csv')
    df_test_raw = pd.read_csv('data/raw/test.csv')
    
    cv = get_cv_strategy(n_splits=5, random_state=42)
    models = get_baseline_models(random_state=42)
    
    target_col = 'digital_academic_wellbeing_level'
    label_map = {'Low': 0, 'Medium': 1, 'High': 2}
    inv_label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    
    # Optuna weights from Phase 10 Experiment
    ensemble_weights = {
        'LogisticRegression': 0.2803,
        'RandomForest': 0.0479,
        'ExtraTrees': 0.1429,
        'MLPClassifier': 0.1588,
        'LightGBM_s1': 0.0450,
        'LightGBM_s2': 0.0542,
        'LightGBM_s3': 0.0942,
        'XGBoost_s1': 0.0207,
        'XGBoost_s2': 0.0621,
        'XGBoost_s3': 0.0241,
        'CatBoost_s1': 0.0010,
        'CatBoost_s2': 0.0687
    }
    
    # --- Step 1: Run CV to get OOF probabilities for threshold optimization ---
    print("\n--- Step 1: Running CV to determine optimal threshold multipliers ---")
    oof_probas_dict = {name: np.zeros((len(df_raw), 3)) for name in models.keys()}
    y_true_all = np.zeros(len(df_raw))
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(df_raw, df_raw[target_col])):
        print(f"  Processing Fold {fold+1}...")
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
        
        for model_name, model in models.items():
            pipeline = build_baseline_pipeline(model, numeric_features, categorical_features)
            if "XGBoost" in model_name:
                sw_train = compute_xgb_sample_weights(y_train)
                pipeline.fit(X_train, y_train, classifier__sample_weight=sw_train)
            else:
                pipeline.fit(X_train, y_train)
            
            oof_probas_dict[model_name][val_idx] = pipeline.predict_proba(X_val)
        
        y_true_all[val_idx] = y_val

    # Blend OOF probabilities
    blended_oof_proba = np.zeros((len(df_raw), 3))
    for name in models.keys():
        blended_oof_proba += oof_probas_dict[name] * ensemble_weights[name]
        
    y_series = pd.Series(y_true_all)
    best_f1, best_multipliers, _ = optimize_thresholds(y_series, blended_oof_proba)
    print(f"\nOptimal Ensemble Threshold Multipliers: {best_multipliers}")
    print(f"CV Macro-F1 with these multipliers: {best_f1:.4f}")

    # --- Step 2: Train on FULL dataset and predict on TEST ---
    print("\n--- Step 2: Training on Full Dataset & Predicting ---")
    df_train_full = apply_canonicalization(df_raw)
    df_test = apply_canonicalization(df_test_raw)
    
    # Feature engineering (test uses train_full as stats_df)
    df_feat_train_full = generate_features(df_train_full)
    df_feat_test = generate_features(df_test, stats_df=df_feat_train_full)
    
    y_train_full = df_feat_train_full[target_col].map(label_map)
    
    X_train_full = df_feat_train_full.drop(columns=cols_to_drop, errors='ignore')
    X_test = df_feat_test.drop(columns=[c for c in cols_to_drop if c != target_col], errors='ignore')
    
    for c in X_train_full.columns:
        if c not in X_test.columns:
            X_test[c] = 0
    X_test = X_test[X_train_full.columns]
    
    test_probas_dict = {}
    for model_name, model in models.items():
        print(f"  Training {model_name} on full dataset...")
        pipeline = build_baseline_pipeline(model, numeric_features, categorical_features)
        
        if "XGBoost" in model_name:
            sw_train = compute_xgb_sample_weights(y_train_full)
            pipeline.fit(X_train_full, y_train_full, classifier__sample_weight=sw_train)
        else:
            pipeline.fit(X_train_full, y_train_full)
            
        test_probas_dict[model_name] = pipeline.predict_proba(X_test)
        
    # Blend test probabilities
    blended_test_proba = np.zeros((len(df_test), 3))
    for name in models.keys():
        blended_test_proba += test_probas_dict[name] * ensemble_weights[name]
        
    # Apply threshold multipliers
    w_test_proba = blended_test_proba * best_multipliers
    test_preds = np.argmax(w_test_proba, axis=1)
    
    # Save submission
    sub = pd.DataFrame({
        'respondent_id': df_test_raw['respondent_id'],
        'digital_academic_wellbeing_level': [inv_label_map[p] for p in test_preds]
    })
    sub_path = 'submission-phase-10.csv'
    sub.to_csv(sub_path, index=False)
    print(f"\nSubmission saved to {sub_path}!")

if __name__ == '__main__':
    run()
