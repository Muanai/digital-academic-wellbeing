import sys
import json
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.preprocessing import apply_canonicalization
from src.features import generate_features, get_model_specific_features
from src.models import get_baseline_models, build_baseline_pipeline
from src.utils import seed_everything

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

def load_best_params():
    try:
        with open('experiments/phase12_best_params.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: phase12_best_params.json not found!")
        sys.exit(1)

def run():
    seed_everything(42)
    print("Loading data...")
    df_raw = pd.read_csv('data/raw/train.csv')
    df_test_raw = pd.read_csv('data/raw/test.csv')
    best_params = load_best_params()
    
    target_col = 'digital_academic_wellbeing_level'
    label_map = {'Low': 0, 'Medium': 1, 'High': 2}
    inv_label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    
    print("Applying Canonicalization...")
    df_train = apply_canonicalization(df_raw)
    df_test = apply_canonicalization(df_test_raw)
    
    # ---------------------------------------------------------
    # Golden Models (Ablated Features) setup
    # ---------------------------------------------------------
    cols_to_drop_train = ['respondent_id', target_col]
    cols_to_drop_test = ['respondent_id']
    raw_to_drop = [c.replace('_clean', '') for c in df_train.columns if c.endswith('_clean')]
    cols_to_drop_train.extend(raw_to_drop)
    cols_to_drop_test.extend(raw_to_drop)
    
    y_train_full = df_train[target_col].map(label_map).values
    
    # Optuna Weights from Phase 13 CV
    ensemble_weights = {
      'XGBoost_Tuned': 0.1476,
      'RandomForest': 0.1307,
      'ExtraTrees': 0.1279,
      'CatBoost_s2': 0.1081,
      'MLPClassifier': 0.0985,
      'CatBoost_Tuned': 0.0816,
      'LightGBM_s3': 0.0780,
      'CatBoost_s1': 0.0626,
      'LogisticRegression': 0.0589,
      'LightGBM_Tuned': 0.0533,
      'LightGBM_s2': 0.0342,
      'LightGBM_s1': 0.0087,
      'XGBoost_s2': 0.0081,
      'XGBoost_s3': 0.0014,
      'XGBoost_s1': 0.0003
    }
    
    best_thresholds = np.array([0.99301644, 1.00928659, 1.00442333])
    
    golden_models = get_baseline_models(random_state=42)
    
    tuned_models = {
        "LightGBM_Tuned": lgb.LGBMClassifier(**best_params['LightGBM']['params'], class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1),
        "XGBoost_Tuned": xgb.XGBClassifier(**best_params['XGBoost']['params'], random_state=42, eval_metric='mlogloss', n_jobs=-1),
        "CatBoost_Tuned": CatBoostClassifier(**best_params['CatBoost']['params'], auto_class_weights='Balanced', random_state=42, verbose=False, thread_count=-1)
    }
    
    all_models = {**golden_models, **tuned_models}
    
    # We will accumulate test probabilities
    test_proba_blend = np.zeros((len(df_test), 3))
    
    # ---------------------------------------------------------
    # Train and Predict Golden Models (Ablated)
    # ---------------------------------------------------------
    print("\n--- Training Golden Models (Phase 6) on 100% Data ---")
    for model_name, model in golden_models.items():
        if ensemble_weights.get(model_name, 0) == 0:
            continue
            
        print(f"  > Training {model_name}...")
        model_type = "tree" if any(t in model_name for t in ["CatBoost", "XGBoost", "LightGBM", "RandomForest", "ExtraTrees"]) else "linear"
        
        df_feat_tr = get_model_specific_features(df_train, model_type=model_type)
        df_feat_te = get_model_specific_features(df_test, model_type=model_type)
        
        X_tr = df_feat_tr.drop(columns=cols_to_drop_train, errors='ignore')
        X_te = df_feat_te.drop(columns=cols_to_drop_test, errors='ignore')
        
        # Align columns
        for c in X_tr.columns:
            if c not in X_te.columns:
                X_te[c] = 0
        X_te = X_te[X_tr.columns]
        
        numeric_features = X_tr.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        categorical_features = [c for c in X_tr.columns if c not in numeric_features]
        
        pipeline = build_baseline_pipeline(model, numeric_features, categorical_features)
        
        if "XGBoost" in model_name:
            from src.models import compute_xgb_sample_weights
            sw = compute_xgb_sample_weights(pd.Series(y_train_full))
            pipeline.fit(X_tr, y_train_full, classifier__sample_weight=sw)
        else:
            pipeline.fit(X_tr, y_train_full)
            
        preds_proba = pipeline.predict_proba(X_te)
        test_proba_blend += preds_proba * ensemble_weights[model_name]
        
    # ---------------------------------------------------------
    # Train and Predict Tuned Models (Full Features)
    # ---------------------------------------------------------
    print("\n--- Training Tuned Models (Phase 12) on 100% Data ---")
    df_feat_full_tr = generate_features(df_train)
    df_feat_full_te = generate_features(df_test, stats_df=df_feat_full_tr)
    
    X_full_tr = df_feat_full_tr.drop(columns=cols_to_drop_train, errors='ignore')
    X_full_te = df_feat_full_te.drop(columns=cols_to_drop_test, errors='ignore')
    
    for c in X_full_tr.columns:
        if c not in X_full_te.columns:
            X_full_te[c] = 0
    X_full_te = X_full_te[X_full_tr.columns]
    
    num_feat_full = X_full_tr.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    cat_feat_full = [c for c in X_full_tr.columns if c not in num_feat_full]
    
    # Preprocess
    from scripts.exp_phase12_hyperparameter_tuning import build_preprocessor
    preprocessor = build_preprocessor(num_feat_full, cat_feat_full)
    X_train_proc = preprocessor.fit_transform(X_full_tr)
    X_test_proc = preprocessor.transform(X_full_te)
    
    for model_name, model in tuned_models.items():
        if ensemble_weights.get(model_name, 0) == 0:
            continue
            
        print(f"  > Training {model_name}...")
        if "XGBoost" in model_name:
            from src.models import compute_xgb_sample_weights
            sw = compute_xgb_sample_weights(pd.Series(y_train_full))
            model.fit(X_train_proc, y_train_full, sample_weight=sw)
        else:
            model.fit(X_train_proc, y_train_full)
            
        preds_proba = model.predict_proba(X_test_proc)
        test_proba_blend += preds_proba * ensemble_weights[model_name]
        
    # ---------------------------------------------------------
    # Apply Thresholds & Save
    # ---------------------------------------------------------
    print("\n--- Applying Optuna Weights and Nelder-Mead Thresholds ---")
    
    # Apply threshold multipliers
    test_proba_blend = test_proba_blend * best_thresholds
    
    final_preds = np.argmax(test_proba_blend, axis=1)
    final_preds_labels = [inv_label_map[p] for p in final_preds]
    
    submission = pd.DataFrame({
        'respondent_id': df_test_raw['respondent_id'],
        'digital_academic_wellbeing_level': final_preds_labels
    })
    
    out_file = 'submission-phase13.csv'
    submission.to_csv(out_file, index=False)
    print(f"\n✅ Submission successfully saved to {out_file}")
    
    # Print label distribution
    dist = submission['digital_academic_wellbeing_level'].value_counts(normalize=True) * 100
    print("\nPredicted Class Distribution:")
    for label, pct in dist.items():
        print(f"  {label}: {pct:.1f}%")

if __name__ == '__main__':
    run()
