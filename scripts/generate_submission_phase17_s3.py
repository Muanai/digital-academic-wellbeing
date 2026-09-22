import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.preprocessing import apply_canonicalization
from src.features import generate_features, add_winning_domain_features
from src.models import compute_xgb_sample_weights
from src.utils import seed_everything

def build_preprocessor(numeric_features, categorical_features):
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
    print("Loading data...")
    df_train_raw = pd.read_csv('data/raw/train.csv')
    df_test_raw = pd.read_csv('data/raw/test.csv')
    
    test_ids = df_test_raw['respondent_id']
    
    target_col = 'digital_academic_wellbeing_level'
    label_map = {'Low': 0, 'Medium': 1, 'High': 2}
    inv_label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    y_train = df_train_raw[target_col].map(label_map).values
    
    # 1. Prepare Full Features (for Boosters)
    print("Preparing Full Features...")
    df_train_canon = apply_canonicalization(df_train_raw)
    df_test_canon = apply_canonicalization(df_test_raw)
    
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
    
    num_features_full = X_train_full.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    cat_features_full = [c for c in X_train_full.columns if c not in num_features_full]
    
    prep_full = build_preprocessor(num_features_full, cat_features_full)
    X_train_full_proc = prep_full.fit_transform(X_train_full)
    X_test_full_proc = prep_full.transform(X_test_full)
    
    # 2. Prepare Winning Domain Features (for RF/ET)
    print("Preparing Winning Domain Features...")
    df_train_wd = add_winning_domain_features(df_train_full)
    df_test_wd = add_winning_domain_features(df_test_full)
    
    X_train_wd = df_train_wd.drop(columns=cols_to_drop, errors='ignore')
    X_test_wd = df_test_wd.drop(columns=[c for c in cols_to_drop if c != target_col], errors='ignore')
    
    for c in X_train_wd.columns:
        if c not in X_test_wd.columns:
            X_test_wd[c] = 0
    X_test_wd = X_test_wd[X_train_wd.columns]
    
    num_features_wd = X_train_wd.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    cat_features_wd = [c for c in X_train_wd.columns if c not in num_features_wd]
    
    prep_wd = build_preprocessor(num_features_wd, cat_features_wd)
    X_train_wd_proc = prep_wd.fit_transform(X_train_wd)
    X_test_wd_proc = prep_wd.transform(X_test_wd)
    
    # 3. Load Phase 12 Params for Boosters
    print("Loading Phase 12 params for Boosters...")
    import json
    with open('experiments/phase12_best_params.json', 'r') as f:
        best_params = json.load(f)
        
    cb_params = best_params['CatBoost']['params'].copy()
    cb_params.update({'auto_class_weights': 'Balanced', 'random_state': 42, 'verbose': False, 'thread_count': -1})
    
    xgb_params = best_params['XGBoost']['params'].copy()
    xgb_params.update({'random_state': 42, 'eval_metric': 'mlogloss', 'n_jobs': -1})
    
    lgb_params = best_params['LightGBM']['params'].copy()
    lgb_params.update({'class_weight': 'balanced', 'random_state': 42, 'verbose': -1, 'n_jobs': -1})
    
    rf_params = {'n_estimators': 300, 'max_depth': 10, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}
    et_params = {'n_estimators': 300, 'max_depth': 10, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}
    
    models_full = {
        'CatBoost_s42': CatBoostClassifier(**cb_params),
        'XGBoost_s42': xgb.XGBClassifier(**xgb_params),
        'LightGBM': lgb.LGBMClassifier(**lgb_params)
    }
    
    models_wd = {
        'RandomForest_WD': RandomForestClassifier(**rf_params),
        'ExtraTrees_WD': ExtraTreesClassifier(**et_params)
    }
    
    sw_train = compute_xgb_sample_weights(y_train)
    
    test_probas = {}
    
    print("Training Boosters on Full Data (Full Features)...")
    for name, model in models_full.items():
        print(f"  > {name}")
        if 'XGBoost' in name:
            model.fit(X_train_full_proc, y_train, sample_weight=sw_train)
        else:
            model.fit(X_train_full_proc, y_train)
        test_probas[name] = model.predict_proba(X_test_full_proc)
        
    print("Training RF/ET on Full Data (Winning Domain Features)...")
    for name, model in models_wd.items():
        print(f"  > {name}")
        model.fit(X_train_wd_proc, y_train)
        test_probas[name] = model.predict_proba(X_test_wd_proc)
        
    print("Blending probabilities...")
    weights = {
        'CatBoost_s42': 0.378651,
        'LightGBM': 0.301925,
        'RandomForest_WD': 0.250290,
        'ExtraTrees_WD': 0.034789,
        'XGBoost_s42': 0.034345
    }
    
    blended = np.zeros((len(test_ids), 3))
    for name, w in weights.items():
        blended += w * test_probas[name]
        
    multipliers = np.array([1.044488, 1.054975, 0.898251])
    adjusted = blended * multipliers
    
    final_preds = np.argmax(adjusted, axis=1)
    final_labels = [inv_label_map[p] for p in final_preds]
    
    submission = pd.DataFrame({
        'respondent_id': test_ids,
        'digital_academic_wellbeing_level': final_labels
    })
    
    submission.to_csv('submission_phase17_s3.csv', index=False)
    print("✅ Created submission_phase17_s3.csv")
    print(submission['digital_academic_wellbeing_level'].value_counts(normalize=True))

if __name__ == '__main__':
    main()
