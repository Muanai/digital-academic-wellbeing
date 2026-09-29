import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from src.preprocessing import apply_canonicalization
from src.features import generate_features
from src.models import compute_xgb_sample_weights
from src.utils import seed_everything

def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', QuantileTransformer(output_distribution='normal', random_state=42))
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

def main():
    seed_everything(42)
    print("=" * 65)
    print("PHASE 18: ANTI-OVERFIT SUBMISSIONS")
    print("=" * 65)
    
    # 1. Load Data
    df_train_raw = pd.read_csv('data/raw/train.csv')
    df_test_raw = pd.read_csv('data/raw/test.csv')
    
    target_col = 'digital_academic_wellbeing_level'
    label_map = {'Low': 0, 'Medium': 1, 'High': 2}
    inv_label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    
    y_true = df_train_raw[target_col].map(label_map).values
    test_ids = df_test_raw['respondent_id']
    
    # 2. Canonicalization
    df_train_canon = apply_canonicalization(df_train_raw)
    df_test_canon = apply_canonicalization(df_test_raw)
    
    # 3. Features (Full)
    print("Preparing Full features for full dataset & test set...")
    df_train_full = generate_features(df_train_canon)
    df_test_full = generate_features(df_test_canon, stats_df=df_train_full)
    
    cols_to_drop = ['respondent_id', target_col]
    raw_to_drop = [c.replace('_clean', '') for c in df_train_full.columns if c.endswith('_clean')]
    cols_to_drop.extend(raw_to_drop)
    
    X_train_full = df_train_full.drop(columns=cols_to_drop, errors='ignore')
    X_test_full = df_test_full.drop(columns=[c for c in cols_to_drop if c != target_col], errors='ignore')
    
    for c in X_train_full.columns:
        if c not in X_test_full.columns:
            X_test_full[c] = 0
    X_test_full = X_test_full[X_train_full.columns]
    
    num_f_full = X_train_full.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    cat_f_full = [c for c in X_train_full.columns if c not in num_f_full]
    
    prep_full = build_preprocessor(num_f_full, cat_f_full)
    X_train_proc = prep_full.fit_transform(X_train_full)
    X_test_proc = prep_full.transform(X_test_full)
    
    sw_full_train = compute_xgb_sample_weights(y_true)
    
    # 4. Load Params
    with open('experiments/phase17_best_params.json', 'r') as f:
        phase17_params = json.load(f)
        
    test_predictions_dict = {}
    
    print("\n--- Training Models on 100% Data ---")
    
    # CatBoost
    print("  Fitting CatBoost...")
    cb_params = phase17_params['CatBoost']['params'].copy()
    cb_params.update({'auto_class_weights': 'Balanced', 'random_state': 42, 'thread_count': -1, 'verbose': False})
    m_cb = CatBoostClassifier(**cb_params)
    m_cb.fit(X_train_proc, y_true)
    test_predictions_dict['CatBoost'] = m_cb.predict_proba(X_test_proc)
    
    # XGBoost
    print("  Fitting XGBoost...")
    xgb_params = phase17_params['XGBoost']['params'].copy()
    xgb_params.update({'random_state': 42, 'n_jobs': -1, 'eval_metric': 'mlogloss'})
    m_xgb = xgb.XGBClassifier(**xgb_params)
    m_xgb.fit(X_train_proc, y_true, sample_weight=sw_full_train)
    test_predictions_dict['XGBoost'] = m_xgb.predict_proba(X_test_proc)
    
    # LightGBM
    print("  Fitting LightGBM...")
    lgb_params = phase17_params['LightGBM']['params'].copy()
    lgb_params.update({'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1, 'verbose': -1})
    m_lgb = lgb.LGBMClassifier(**lgb_params)
    m_lgb.fit(X_train_proc, y_true)
    test_predictions_dict['LightGBM'] = m_lgb.predict_proba(X_test_proc)
    
    # --- STRATEGY 1: Raw Equal Ensemble ---
    print("\n[Strategy 1] Generating submission_raw_equal.csv...")
    test_blended_equal = (test_predictions_dict['CatBoost'] + 
                          test_predictions_dict['XGBoost'] + 
                          test_predictions_dict['LightGBM']) / 3.0
                          
    preds_equal = np.argmax(test_blended_equal, axis=1)
    sub_equal = pd.DataFrame({
        'respondent_id': test_ids,
        'digital_academic_wellbeing_level': [inv_label_map[p] for p in preds_equal]
    })
    sub_equal.to_csv('submission_raw_equal.csv', index=False)
    print("Test Prediction Class Distribution (Strategy 1 - Equal Ensemble):")
    print(sub_equal['digital_academic_wellbeing_level'].value_counts(normalize=True).round(4) * 100)
    
    # --- STRATEGY: CatBoost Solo Raw ---
    print("\n[Strategy] Generating submission_raw_catboost_solo.csv...")
    preds_cb = np.argmax(test_predictions_dict['CatBoost'], axis=1)
    sub_cb = pd.DataFrame({
        'respondent_id': test_ids,
        'digital_academic_wellbeing_level': [inv_label_map[p] for p in preds_cb]
    })
    sub_cb.to_csv('submission_raw_catboost_solo.csv', index=False)
    print("Test Prediction Class Distribution (CatBoost Solo):")
    print(sub_cb['digital_academic_wellbeing_level'].value_counts(normalize=True).round(4) * 100)

    print("\n✅ Submissions created successfully.")

if __name__ == '__main__':
    main()
