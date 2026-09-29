"""
Phase 18B: Multi-Strategy Anti-Overfit Submission Generator
===========================================================
Generates multiple submission files using different anti-overfitting strategies:

1. Multi-Seed Test Averaging (same model, different seeds -> average test preds)
2. Feature-Reduced Model (fewer features to reduce overfitting)
3. Regularized Ensemble (stronger regularization on all 3 boosters)
4. Stacking with Regularized Meta-Learner (LR on OOF probas)
5. All combined: Multi-seed + Regularized + Stacking
"""

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from src.preprocessing import apply_canonicalization
from src.features import generate_features
from src.models import compute_xgb_sample_weights
from src.utils import seed_everything
from src.validation import optimize_thresholds


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


def prepare_data(df_train_raw, df_test_raw, target_col, use_full_features=True):
    """Prepare train and test data with canonicalization and feature engineering."""
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
    
    return X_train_proc, X_test_proc, num_f, cat_f


def train_catboost(X_train, y_train, params, seed=42):
    p = params.copy()
    p.update({'auto_class_weights': 'Balanced', 'random_state': seed, 'thread_count': -1, 'verbose': False})
    m = CatBoostClassifier(**p)
    m.fit(X_train, y_train)
    return m


def train_xgboost(X_train, y_train, sw, params, seed=42):
    p = params.copy()
    p.update({'random_state': seed, 'n_jobs': -1, 'eval_metric': 'mlogloss'})
    m = xgb.XGBClassifier(**p)
    m.fit(X_train, y_train, sample_weight=sw)
    return m


def train_lightgbm(X_train, y_train, params, seed=42):
    p = params.copy()
    p.update({'class_weight': 'balanced', 'random_state': seed, 'n_jobs': -1, 'verbose': -1})
    m = lgb.LGBMClassifier(**p)
    m.fit(X_train, y_train)
    return m


def save_submission(test_ids, preds_numeric, inv_label_map, filename, label):
    labels = [inv_label_map[p] for p in preds_numeric]
    sub = pd.DataFrame({
        'respondent_id': test_ids,
        'digital_academic_wellbeing_level': labels
    })
    sub.to_csv(filename, index=False)
    dist = sub['digital_academic_wellbeing_level'].value_counts(normalize=True).round(4) * 100
    print(f"  [{label}] Distribution: Low={dist.get('Low',0):.1f}%, Med={dist.get('Medium',0):.1f}%, High={dist.get('High',0):.1f}%")
    return sub


def main():
    seed_everything(42)
    print("=" * 65)
    print("PHASE 18B: MULTI-STRATEGY ANTI-OVERFIT SUBMISSIONS")
    print("=" * 65)
    
    # Load
    df_train_raw = pd.read_csv('data/raw/train.csv')
    df_test_raw = pd.read_csv('data/raw/test.csv')
    target_col = 'digital_academic_wellbeing_level'
    label_map = {'Low': 0, 'Medium': 1, 'High': 2}
    inv_label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    y_true = df_train_raw[target_col].map(label_map).values
    test_ids = df_test_raw['respondent_id']
    
    with open('experiments/phase17_best_params.json', 'r') as f:
        phase17_params = json.load(f)
    
    # Prepare data
    print("\nPreparing data...")
    X_train_proc, X_test_proc, num_f, cat_f = prepare_data(
        df_train_raw, df_test_raw, target_col
    )
    sw = compute_xgb_sample_weights(y_true)
    
    cb_params = phase17_params['CatBoost']['params']
    xgb_params = phase17_params['XGBoost']['params']
    lgb_params = phase17_params['LightGBM']['params']
    
    # =========================================================================
    # STRATEGY A: Multi-Seed Test Averaging (TTA for Tabular)
    # Train each model with 5 different seeds, average test predictions.
    # This reduces variance in test predictions without overfitting to OOF.
    # =========================================================================
    print("\n--- STRATEGY A: Multi-Seed Test Averaging ---")
    seeds = [42, 123, 456, 789, 1024]
    
    test_probas_cb = []
    test_probas_xgb = []
    test_probas_lgb = []
    
    for s in seeds:
        print(f"  Training seed={s}...", flush=True)
        m_cb = train_catboost(X_train_proc, y_true, cb_params, seed=s)
        m_xgb = train_xgboost(X_train_proc, y_true, sw, xgb_params, seed=s)
        m_lgb = train_lightgbm(X_train_proc, y_true, lgb_params, seed=s)
        
        test_probas_cb.append(m_cb.predict_proba(X_test_proc))
        test_probas_xgb.append(m_xgb.predict_proba(X_test_proc))
        test_probas_lgb.append(m_lgb.predict_proba(X_test_proc))
    
    avg_cb = np.mean(test_probas_cb, axis=0)
    avg_xgb = np.mean(test_probas_xgb, axis=0)
    avg_lgb = np.mean(test_probas_lgb, axis=0)
    
    # A1: Multi-seed equal blend with OOF-tuned thresholds
    blend_multiseed = (avg_cb + avg_xgb + avg_lgb) / 3.0
    
    # Use the OOF thresholds from Phase 17 (they helped on LB)
    with open('experiments/phase17_ensemble_config.json', 'r') as f:
        p17_config = json.load(f)
    mults = np.array(p17_config['threshold_multipliers'])
    
    adjusted = blend_multiseed * mults
    preds_a1 = np.argmax(adjusted, axis=1)
    save_submission(test_ids, preds_a1, inv_label_map,
                    'submission_multiseed_equal_tuned.csv',
                    'A1: MultiSeed EqualW + Thresholds')
    
    # A2: Multi-seed with Phase17 optimized weights + thresholds
    weights_p17 = p17_config['weights']
    total_w = sum(weights_p17.values())
    blend_multiseed_weighted = np.zeros_like(blend_multiseed)
    for name, w in weights_p17.items():
        if name == 'CatBoost':
            blend_multiseed_weighted += (w / total_w) * avg_cb
        elif name == 'XGBoost':
            blend_multiseed_weighted += (w / total_w) * avg_xgb
        elif name == 'LightGBM':
            blend_multiseed_weighted += (w / total_w) * avg_lgb
    
    adjusted2 = blend_multiseed_weighted * mults
    preds_a2 = np.argmax(adjusted2, axis=1)
    save_submission(test_ids, preds_a2, inv_label_map,
                    'submission_multiseed_p17weights_tuned.csv',
                    'A2: MultiSeed P17Weights + Thresholds')
    
    # A3: Multi-seed CatBoost solo with thresholds (CatBoost was strongest)
    cb_mults_raw = np.load('experiments/phase17_oof_probas.npz')
    cb_oof = cb_mults_raw['CatBoost']
    _, cb_solo_mults, _ = optimize_thresholds(pd.Series(y_true), cb_oof)
    
    adjusted3 = avg_cb * cb_solo_mults
    preds_a3 = np.argmax(adjusted3, axis=1)
    save_submission(test_ids, preds_a3, inv_label_map,
                    'submission_multiseed_catboost_tuned.csv',
                    'A3: MultiSeed CatBoost Solo + Thresholds')
    
    # =========================================================================
    # STRATEGY B: Regularized Ensemble 
    # Use much stronger regularization to combat overfitting.
    # =========================================================================
    print("\n--- STRATEGY B: Regularized Ensemble ---")
    
    reg_cb_params = cb_params.copy()
    reg_cb_params['l2_leaf_reg'] = 3.0  # was 0.024
    reg_cb_params['random_strength'] = 5.0  # was 2.15
    
    reg_xgb_params = xgb_params.copy()
    reg_xgb_params['reg_lambda'] = 3.0  # was 0.0005
    reg_xgb_params['reg_alpha'] = 1.0  # was 0.19
    reg_xgb_params['min_child_weight'] = 5  # was 1
    
    reg_lgb_params = lgb_params.copy()
    reg_lgb_params['reg_lambda'] = 3.0  # was 0.003
    reg_lgb_params['reg_alpha'] = 1.0  # was 0.089
    reg_lgb_params['min_child_samples'] = 50  # was 30
    
    # Multi-seed regularized
    reg_cb_probas = []
    reg_xgb_probas = []
    reg_lgb_probas = []
    
    for s in seeds:
        print(f"  Training regularized seed={s}...", flush=True)
        m_cb = train_catboost(X_train_proc, y_true, reg_cb_params, seed=s)
        m_xgb = train_xgboost(X_train_proc, y_true, sw, reg_xgb_params, seed=s)
        m_lgb = train_lightgbm(X_train_proc, y_true, reg_lgb_params, seed=s)
        
        reg_cb_probas.append(m_cb.predict_proba(X_test_proc))
        reg_xgb_probas.append(m_xgb.predict_proba(X_test_proc))
        reg_lgb_probas.append(m_lgb.predict_proba(X_test_proc))
    
    avg_reg_cb = np.mean(reg_cb_probas, axis=0)
    avg_reg_xgb = np.mean(reg_xgb_probas, axis=0)
    avg_reg_lgb = np.mean(reg_lgb_probas, axis=0)
    
    blend_reg = (avg_reg_cb + avg_reg_xgb + avg_reg_lgb) / 3.0
    
    # Also compute OOF for regularized to get thresholds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_reg = np.zeros((len(y_true), 3))
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_proc, y_true)):
        Xtr, Xval = X_train_proc[train_idx], X_train_proc[val_idx]
        ytr, yval = y_true[train_idx], y_true[val_idx]
        sw_tr = sw[train_idx]
        
        m_cb = train_catboost(Xtr, ytr, reg_cb_params, seed=42)
        m_xgb = train_xgboost(Xtr, ytr, sw_tr, reg_xgb_params, seed=42)
        m_lgb = train_lightgbm(Xtr, ytr, reg_lgb_params, seed=42)
        
        p_cb = m_cb.predict_proba(Xval)
        p_xgb = m_xgb.predict_proba(Xval)
        p_lgb = m_lgb.predict_proba(Xval)
        
        oof_reg[val_idx] = (p_cb + p_xgb + p_lgb) / 3.0
    
    oof_raw_f1 = f1_score(y_true, np.argmax(oof_reg, axis=1), average='macro')
    oof_tuned_f1, reg_mults, _ = optimize_thresholds(pd.Series(y_true), oof_reg)
    print(f"  Regularized OOF: Raw F1={oof_raw_f1:.4f}, Tuned F1={oof_tuned_f1:.4f}")
    print(f"  Regularized Thresholds: {reg_mults.round(4)}")
    
    adjusted_reg = blend_reg * reg_mults
    preds_b = np.argmax(adjusted_reg, axis=1)
    save_submission(test_ids, preds_b, inv_label_map,
                    'submission_regularized_tuned.csv',
                    'B: Regularized + MultiSeed + Thresholds')
    
    # =========================================================================
    # STRATEGY C: Stacking with Logistic Regression Meta-Learner
    # Use OOF probabilities from 3 boosters as features for a LR meta-learner.
    # This learns optimal blending in a more principled way than Optuna search.
    # =========================================================================
    print("\n--- STRATEGY C: Stacking Meta-Learner ---")
    
    oof_probas = np.load('experiments/phase17_oof_probas.npz')
    
    # Build meta-features: 3 models x 3 classes = 9 features
    meta_train = np.hstack([
        oof_probas['CatBoost'],
        oof_probas['XGBoost'],
        oof_probas['LightGBM']
    ])
    
    meta_test = np.hstack([avg_cb, avg_xgb, avg_lgb])
    
    # Use nested CV for stacking to avoid leakage
    oof_stack_probas = np.zeros((len(y_true), 3))
    stack_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (tr_idx, val_idx) in enumerate(stack_cv.split(meta_train, y_true)):
        lr = LogisticRegression(C=0.1, max_iter=1000, multi_class='multinomial', random_state=42)
        lr.fit(meta_train[tr_idx], y_true[tr_idx])
        oof_stack_probas[val_idx] = lr.predict_proba(meta_train[val_idx])
    
    stack_raw_f1 = f1_score(y_true, np.argmax(oof_stack_probas, axis=1), average='macro')
    stack_tuned_f1, stack_mults, _ = optimize_thresholds(pd.Series(y_true), oof_stack_probas)
    print(f"  Stacking OOF: Raw F1={stack_raw_f1:.4f}, Tuned F1={stack_tuned_f1:.4f}")
    
    # Train final stacker on all data
    lr_final = LogisticRegression(C=0.1, max_iter=1000, multi_class='multinomial', random_state=42)
    lr_final.fit(meta_train, y_true)
    stack_test_proba = lr_final.predict_proba(meta_test)
    
    adjusted_stack = stack_test_proba * stack_mults
    preds_c = np.argmax(adjusted_stack, axis=1)
    save_submission(test_ids, preds_c, inv_label_map,
                    'submission_stacking_tuned.csv',
                    'C: Stacking LR + Thresholds')
    
    # =========================================================================
    # STRATEGY D: Mega-Blend - Average of ALL strategies
    # Average the probability outputs of Strategies A, B, C.
    # This gives maximum diversity.
    # =========================================================================
    print("\n--- STRATEGY D: Mega-Blend ---")
    
    # Average raw probabilities from all strategies
    mega_blend = (blend_multiseed + blend_reg + stack_test_proba) / 3.0
    
    # Compute OOF for mega-blend to get thresholds
    mega_oof = (oof_reg + oof_stack_probas) / 2.0  # approximate
    _, mega_mults, _ = optimize_thresholds(pd.Series(y_true), mega_oof)
    
    adjusted_mega = mega_blend * mega_mults
    preds_d = np.argmax(adjusted_mega, axis=1)
    save_submission(test_ids, preds_d, inv_label_map,
                    'submission_megablend_tuned.csv',
                    'D: MegaBlend All Strategies')
    
    # Also raw argmax mega-blend
    preds_d_raw = np.argmax(mega_blend, axis=1)
    save_submission(test_ids, preds_d_raw, inv_label_map,
                    'submission_megablend_raw.csv',
                    'D_raw: MegaBlend Raw')
    
    # =========================================================================
    # STRATEGY E: Original Phase17 + Multi-Seed (keep what works, add stability)
    # =========================================================================
    print("\n--- STRATEGY E: Phase17 Config + Multi-Seed Averaging ---")
    # This is the most conservative: keep the exact same config that got 0.66
    # but just add multi-seed averaging for more stable test predictions.
    # The thresholds and weights are from Phase 17 (what got us 0.66).
    
    preds_e = preds_a2  # Already computed above as A2
    print("  (Same as A2 - already generated)")
    
    print("\n" + "=" * 65)
    print("SUMMARY OF ALL SUBMISSIONS")
    print("=" * 65)
    submissions = [
        ('submission_multiseed_equal_tuned.csv', 'A1: MultiSeed + EqualWeights + Thresholds'),
        ('submission_multiseed_p17weights_tuned.csv', 'A2: MultiSeed + P17Weights + Thresholds'),
        ('submission_multiseed_catboost_tuned.csv', 'A3: MultiSeed + CatBoost Solo + Thresholds'),
        ('submission_regularized_tuned.csv', 'B: Regularized + MultiSeed + Thresholds'),
        ('submission_stacking_tuned.csv', 'C: Stacking LR + Thresholds'),
        ('submission_megablend_tuned.csv', 'D: MegaBlend + Thresholds'),
        ('submission_megablend_raw.csv', 'D_raw: MegaBlend Raw Argmax'),
    ]
    
    for fname, label in submissions:
        sub = pd.read_csv(fname)
        dist = sub['digital_academic_wellbeing_level'].value_counts(normalize=True)
        print(f"  {label:50s} L={dist.get('Low',0)*100:.1f}% M={dist.get('Medium',0)*100:.1f}% H={dist.get('High',0)*100:.1f}%")
    
    print("\nDone! Submit these to Kaggle to find the best LB score.")
    print("=" * 65)

if __name__ == '__main__':
    main()
