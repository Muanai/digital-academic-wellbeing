"""
Phase 18B Part 2: Stacking + MegaBlend (continues from Part 1)
Strategy C and D only - A and B already completed.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy.optimize import minimize

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from src.preprocessing import apply_canonicalization
from src.features import generate_features
from src.models import compute_xgb_sample_weights
from src.utils import seed_everything
from src.validation import optimize_thresholds
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


def build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', KNNImputer(n_neighbors=5)),
                              ('scaler', QuantileTransformer(output_distribution='normal', random_state=42))]),
             numeric_features),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                              ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),
             categorical_features)
        ], remainder='drop')


def save_submission(test_ids, preds_numeric, inv_label_map, filename, label):
    labels = [inv_label_map[p] for p in preds_numeric]
    sub = pd.DataFrame({'respondent_id': test_ids, 'digital_academic_wellbeing_level': labels})
    sub.to_csv(filename, index=False)
    dist = sub['digital_academic_wellbeing_level'].value_counts(normalize=True).round(4) * 100
    print(f"  [{label}] Distribution: Low={dist.get('Low',0):.1f}%, Med={dist.get('Medium',0):.1f}%, High={dist.get('High',0):.1f}%")


def main():
    seed_everything(42)
    
    df_train_raw = pd.read_csv('data/raw/train.csv')
    df_test_raw = pd.read_csv('data/raw/test.csv')
    target_col = 'digital_academic_wellbeing_level'
    label_map = {'Low': 0, 'Medium': 1, 'High': 2}
    inv_label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    y_true = df_train_raw[target_col].map(label_map).values
    test_ids = df_test_raw['respondent_id']
    
    with open('experiments/phase17_best_params.json', 'r') as f:
        phase17_params = json.load(f)
    with open('experiments/phase17_ensemble_config.json', 'r') as f:
        p17_config = json.load(f)
    
    # Prepare data
    print("Preparing data...")
    df_train_canon = apply_canonicalization(df_train_raw)
    df_test_canon = apply_canonicalization(df_test_raw)
    df_train_feat = generate_features(df_train_canon)
    df_test_feat = generate_features(df_test_canon, stats_df=df_train_feat)
    
    cols_to_drop = ['respondent_id', target_col]
    raw_to_drop = [c.replace('_clean', '') for c in df_train_feat.columns if c.endswith('_clean')]
    cols_to_drop.extend(raw_to_drop)
    
    X_train = df_train_feat.drop(columns=cols_to_drop, errors='ignore')
    X_test = df_test_feat.drop(columns=[c for c in cols_to_drop if c != target_col], errors='ignore')
    for c in X_train.columns:
        if c not in X_test.columns:
            X_test[c] = 0
    X_test = X_test[X_train.columns]
    
    num_f = X_train.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    cat_f = [c for c in X_train.columns if c not in num_f]
    prep = build_preprocessor(num_f, cat_f)
    X_train_proc = prep.fit_transform(X_train)
    X_test_proc = prep.transform(X_test)
    sw = compute_xgb_sample_weights(y_true)
    
    # Rebuild multi-seed test predictions for the meta-test features
    seeds = [42, 123, 456, 789, 1024]
    cb_params = phase17_params['CatBoost']['params']
    xgb_params = phase17_params['XGBoost']['params']
    lgb_params = phase17_params['LightGBM']['params']
    
    print("Re-training multi-seed models for test meta-features...")
    test_cb, test_xgb, test_lgb = [], [], []
    for s in seeds:
        print(f"  seed={s}...", flush=True)
        p = cb_params.copy(); p.update({'auto_class_weights':'Balanced','random_state':s,'thread_count':-1,'verbose':False})
        m = CatBoostClassifier(**p); m.fit(X_train_proc, y_true)
        test_cb.append(m.predict_proba(X_test_proc))
        
        p = xgb_params.copy(); p.update({'random_state':s,'n_jobs':-1,'eval_metric':'mlogloss'})
        m = xgb.XGBClassifier(**p); m.fit(X_train_proc, y_true, sample_weight=sw)
        test_xgb.append(m.predict_proba(X_test_proc))
        
        p = lgb_params.copy(); p.update({'class_weight':'balanced','random_state':s,'n_jobs':-1,'verbose':-1})
        m = lgb.LGBMClassifier(**p); m.fit(X_train_proc, y_true)
        test_lgb.append(m.predict_proba(X_test_proc))
    
    avg_cb = np.mean(test_cb, axis=0)
    avg_xgb = np.mean(test_xgb, axis=0)
    avg_lgb = np.mean(test_lgb, axis=0)
    
    # =========================================================================
    # STRATEGY C: Stacking with LR Meta-Learner
    # =========================================================================
    print("\n--- STRATEGY C: Stacking Meta-Learner ---")
    oof_probas = np.load('experiments/phase17_oof_probas.npz')
    
    meta_train = np.hstack([oof_probas['CatBoost'], oof_probas['XGBoost'], oof_probas['LightGBM']])
    meta_test = np.hstack([avg_cb, avg_xgb, avg_lgb])
    
    oof_stack_probas = np.zeros((len(y_true), 3))
    stack_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (tr_idx, val_idx) in enumerate(stack_cv.split(meta_train, y_true)):
        lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        lr.fit(meta_train[tr_idx], y_true[tr_idx])
        oof_stack_probas[val_idx] = lr.predict_proba(meta_train[val_idx])
    
    stack_raw_f1 = f1_score(y_true, np.argmax(oof_stack_probas, axis=1), average='macro')
    stack_tuned_f1, stack_mults, _ = optimize_thresholds(pd.Series(y_true), oof_stack_probas)
    print(f"  Stacking OOF: Raw F1={stack_raw_f1:.4f}, Tuned F1={stack_tuned_f1:.4f}")
    
    lr_final = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    lr_final.fit(meta_train, y_true)
    stack_test_proba = lr_final.predict_proba(meta_test)
    
    adjusted_stack = stack_test_proba * stack_mults
    preds_c = np.argmax(adjusted_stack, axis=1)
    save_submission(test_ids, preds_c, inv_label_map,
                    'submission_stacking_tuned.csv', 'C: Stacking LR + Thresholds')
    
    # Also raw stacking
    preds_c_raw = np.argmax(stack_test_proba, axis=1)
    save_submission(test_ids, preds_c_raw, inv_label_map,
                    'submission_stacking_raw.csv', 'C_raw: Stacking LR Raw')
    
    # =========================================================================
    # STRATEGY D: Mega-Blend
    # =========================================================================
    print("\n--- STRATEGY D: Mega-Blend ---")
    
    # Read the already-generated regularized submission to get its distribution
    blend_multiseed = (avg_cb + avg_xgb + avg_lgb) / 3.0
    
    # Combine multi-seed blend + stacking output
    mega_blend = (blend_multiseed + stack_test_proba) / 2.0
    
    mults = np.array(p17_config['threshold_multipliers'])
    adjusted_mega = mega_blend * mults
    preds_d = np.argmax(adjusted_mega, axis=1)
    save_submission(test_ids, preds_d, inv_label_map,
                    'submission_megablend_tuned.csv', 'D: MegaBlend + P17 Thresholds')
    
    preds_d_raw = np.argmax(mega_blend, axis=1)
    save_submission(test_ids, preds_d_raw, inv_label_map,
                    'submission_megablend_raw.csv', 'D_raw: MegaBlend Raw')
    
    print("\nAll done!")


if __name__ == '__main__':
    main()
