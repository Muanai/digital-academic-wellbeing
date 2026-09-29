"""
Phase 17: Deep-Tuned Tree-Based Ensemble Evaluation
===================================================
Evaluates the 5 deep-tuned tree-based models:
  - CatBoost (Tuned, Full Features)
  - XGBoost (Tuned, Full Features)
  - LightGBM (Tuned, Full Features)
  - RandomForest_WD (Tuned, Winning Domain Features)
  - ExtraTrees_WD (Tuned, Winning Domain Features)

Steps:
  1. Generate 5-fold CV OOF probability matrices using optimal params from phase17_best_params.json.
  2. Save OOF arrays to experiments/phase17_oof_probas.npz.
  3. Evaluate individual model performances.
  4. Optimize ensemble weights (Dirichlet/Simplex search via Optuna) with post-hoc Nelder-Mead threshold multipliers.
  5. Test candidate combinations (Boosters-only, All 5 Tree-Based, Diverse Sub-Ensembles).
  6. Log to MLflow under 'Phase17_Deep_Tuning_Ensemble'.
  7. Train on full train data and generate submission_phase17_deep_ensemble.csv.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import optuna
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from src.preprocessing import apply_canonicalization
from src.features import generate_features, add_winning_domain_features
from src.validation import get_cv_strategy, optimize_thresholds
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


def precompute_folds(df_raw, target_col, label_map, cv, feature_mode='full'):
    print(f"Precomputing fold features ({feature_mode}) & preprocessed arrays...", flush=True)
    folds_data = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(df_raw, df_raw[target_col])):
        df_train = df_raw.iloc[train_idx].copy()
        df_val = df_raw.iloc[val_idx].copy()
        
        df_train = apply_canonicalization(df_train)
        df_val = apply_canonicalization(df_val)
        
        if feature_mode == 'winning':
            df_feat_train = generate_features(df_train)
            df_feat_train = add_winning_domain_features(df_feat_train)
            df_feat_val = generate_features(df_val, stats_df=df_feat_train)
            df_feat_val = add_winning_domain_features(df_feat_val)
        else:
            df_feat_train = generate_features(df_train)
            df_feat_val = generate_features(df_val, stats_df=df_feat_train)
            
        y_train = df_feat_train[target_col].map(label_map).values
        y_val = df_feat_val[target_col].map(label_map).values
        
        cols_to_drop = ['respondent_id', target_col]
        raw_to_drop = [c.replace('_clean', '') for c in df_feat_train.columns if c.endswith('_clean')]
        cols_to_drop.extend(raw_to_drop)
        
        X_train = df_feat_train.drop(columns=cols_to_drop, errors='ignore')
        X_val = df_feat_val.drop(columns=cols_to_drop, errors='ignore')
        
        for c in X_train.columns:
            if c not in X_val.columns:
                X_val[c] = 0
        X_val = X_val[X_train.columns]
        
        numeric_features = X_train.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        categorical_features = [c for c in X_train.columns if c not in numeric_features]
        
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)
        
        sw_train = compute_xgb_sample_weights(y_train)
        
        folds_data.append({
            'X_train': X_train_proc, 'y_train': y_train,
            'X_val': X_val_proc, 'y_val': y_val,
            'val_idx': val_idx,
            'sw_train': sw_train
        })
        
    return folds_data


def generate_oof_probas(model_family, params, folds_data, len_df):
    oof_proba = np.zeros((len_df, 3))
    
    for fold, fd in enumerate(folds_data):
        if model_family == 'LightGBM':
            m_params = params.copy()
            m_params.update({'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1, 'verbose': -1})
            model = lgb.LGBMClassifier(**m_params)
            model.fit(fd['X_train'], fd['y_train'])
            
        elif model_family == 'XGBoost':
            m_params = params.copy()
            m_params.update({'random_state': 42, 'n_jobs': -1, 'eval_metric': 'mlogloss'})
            model = xgb.XGBClassifier(**m_params)
            model.fit(fd['X_train'], fd['y_train'], sample_weight=fd['sw_train'])
            
        elif model_family == 'CatBoost':
            m_params = params.copy()
            m_params.update({'auto_class_weights': 'Balanced', 'random_state': 42, 'thread_count': -1, 'verbose': False})
            model = CatBoostClassifier(**m_params)
            model.fit(fd['X_train'], fd['y_train'])
            
        elif model_family == 'RandomForest':
            m_params = params.copy()
            m_params.update({'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1})
            model = RandomForestClassifier(**m_params)
            model.fit(fd['X_train'], fd['y_train'])
            
        elif model_family == 'ExtraTrees':
            m_params = params.copy()
            m_params.update({'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1})
            model = ExtraTreesClassifier(**m_params)
            model.fit(fd['X_train'], fd['y_train'])
            
        else:
            raise ValueError(f"Unknown model_family: {model_family}")
            
        proba = model.predict_proba(fd['X_val'])
        oof_proba[fd['val_idx']] = proba
        
    return oof_proba


def main():
    seed_everything(42)
    print("=" * 65)
    print("PHASE 17: DEEP-TUNED TREE-BASED ENSEMBLE EVALUATION")
    print("=" * 65)
    
    # 1. Load raw train data
    df_raw = pd.read_csv('data/raw/train.csv')
    target_col = 'digital_academic_wellbeing_level'
    label_map = {'Low': 0, 'Medium': 1, 'High': 2}
    inv_label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    y_true = df_raw[target_col].map(label_map).values
    len_df = len(df_raw)
    
    cv = get_cv_strategy(n_splits=5, random_state=42)
    
    # 2. Precompute fold data
    folds_data_full = precompute_folds(df_raw, target_col, label_map, cv, feature_mode='full')
    folds_data_winning = precompute_folds(df_raw, target_col, label_map, cv, feature_mode='winning')
    
    # 3. Load Phase 17 best parameters
    with open('experiments/phase17_best_params.json', 'r') as f:
        phase17_params = json.load(f)
        
    print("\n--- Generating OOF Probabilities for All 5 Models ---")
    models_config = {
        'CatBoost': ('CatBoost', phase17_params['CatBoost']['params'], folds_data_full),
        'XGBoost': ('XGBoost', phase17_params['XGBoost']['params'], folds_data_full),
        'LightGBM': ('LightGBM', phase17_params['LightGBM']['params'], folds_data_full),
        'RandomForest_WD': ('RandomForest', phase17_params['RandomForest_WD']['params'], folds_data_winning),
        'ExtraTrees_WD': ('ExtraTrees', phase17_params['ExtraTrees_WD']['params'], folds_data_winning)
    }
    
    oof_probas = {}
    for name, (family, params, folds) in models_config.items():
        print(f"  > Training {name} across 5 folds...", flush=True)
        oof_probas[name] = generate_oof_probas(family, params, folds, len_df)
        
    # Save OOF probas for reproducibility & reuse
    np.savez('experiments/phase17_oof_probas.npz', **oof_probas)
    print("✅ Saved OOF probabilities to experiments/phase17_oof_probas.npz")
    
    # 4. Evaluate Individual Models
    print("\n" + "=" * 65)
    print("INDIVIDUAL MODEL PERFORMANCE (5-Fold CV OOF)")
    print("=" * 65)
    print(f"{'Model':20s} {'Raw Macro-F1':15s} {'Tuned Macro-F1':15s} {'Multipliers'}")
    print("-" * 65)
    
    single_results = {}
    for name, proba in oof_probas.items():
        raw_f1 = f1_score(y_true, np.argmax(proba, axis=1), average='macro')
        tuned_f1, multipliers, _ = optimize_thresholds(pd.Series(y_true), proba)
        mult_str = f"[{multipliers[0]:.3f}, {multipliers[1]:.3f}, {multipliers[2]:.3f}]"
        print(f"{name:20s} {raw_f1:<15.4f} {tuned_f1:<15.4f} {mult_str}")
        single_results[name] = {'raw_f1': raw_f1, 'tuned_f1': tuned_f1, 'multipliers': multipliers}
        
    # 5. Evaluate Candidate Ensembles
    print("\n" + "=" * 65)
    print("ENSEMBLE WEIGHT OPTIMIZATION")
    print("=" * 65)
    
    def evaluate_blend(weights_dict, apply_threshold_tuning=True):
        blended = np.zeros((len_df, 3))
        total_w = sum(weights_dict.values())
        for m_name, w in weights_dict.items():
            blended += (w / total_w) * oof_probas[m_name]
            
        raw_f1 = f1_score(y_true, np.argmax(blended, axis=1), average='macro')
        if apply_threshold_tuning:
            tuned_f1, multipliers, preds = optimize_thresholds(pd.Series(y_true), blended)
            return raw_f1, tuned_f1, multipliers, preds, blended
        return raw_f1, raw_f1, np.array([1.0, 1.0, 1.0]), np.argmax(blended, axis=1), blended

    # Scenario A: Equal Weights (All 5)
    equal_weights_5 = {k: 1.0 / len(oof_probas) for k in oof_probas}
    eq_raw, eq_tuned, eq_mult, _, _ = evaluate_blend(equal_weights_5)
    print(f"Equal Weights (All 5): Raw F1 = {eq_raw:.4f} | Tuned F1 = {eq_tuned:.4f} | Mult: {eq_mult.round(3)}")

    # Scenario B: Equal Weights (Boosters only: CB, XGB, LGB)
    equal_weights_boosters = {'CatBoost': 1/3, 'XGBoost': 1/3, 'LightGBM': 1/3}
    eqb_raw, eqb_tuned, eqb_mult, _, _ = evaluate_blend(equal_weights_boosters)
    print(f"Equal Weights (Boosters): Raw F1 = {eqb_raw:.4f} | Tuned F1 = {eqb_tuned:.4f} | Mult: {eqb_mult.round(3)}")

    # Scenario C: Optuna Weight Search (All 5 Tree-Based Models)
    print("\nRunning Optuna to optimize ensemble weights for all 5 models (500 trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    all_model_names = list(oof_probas.keys())
    
    def objective_ensemble(trial):
        raw_weights = {name: trial.suggest_float(f"w_{name}", 0.0, 1.0) for name in all_model_names}
        total_w = sum(raw_weights.values())
        if total_w == 0:
            return 0.0
        normalized_w = {k: v / total_w for k, v in raw_weights.items()}
        
        blended = np.zeros((len_df, 3))
        for m_name, w in normalized_w.items():
            blended += w * oof_probas[m_name]
            
        tuned_f1, _, _ = optimize_thresholds(pd.Series(y_true), blended)
        return tuned_f1 if tuned_f1 is not None else 0.0

    study_ensemble = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study_ensemble.optimize(objective_ensemble, n_trials=500)
    
    best_raw_w = {name: study_ensemble.best_params[f"w_{name}"] for name in all_model_names}
    best_norm_w = {name: best_raw_w[name] / sum(best_raw_w.values()) for name in all_model_names}
    
    opt_raw, opt_tuned, opt_mult, opt_preds, opt_blended = evaluate_blend(best_norm_w)
    
    # Scenario D: Boosters Only Optuna Weight Search
    print("Running Optuna for Boosters Only (CatBoost, XGBoost, LightGBM) (300 trials)...")
    booster_names = ['CatBoost', 'XGBoost', 'LightGBM']
    def objective_boosters(trial):
        raw_w = {name: trial.suggest_float(f"w_{name}", 0.0, 1.0) for name in booster_names}
        total_w = sum(raw_w.values())
        if total_w == 0:
            return 0.0
        norm_w = {k: v / total_w for k, v in raw_w.items()}
        blended = np.zeros((len_df, 3))
        for m_name, w in norm_w.items():
            blended += w * oof_probas[m_name]
        tuned_f1, _, _ = optimize_thresholds(pd.Series(y_true), blended)
        return tuned_f1 if tuned_f1 is not None else 0.0
        
    study_boosters = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study_boosters.optimize(objective_boosters, n_trials=300)
    best_booster_raw = {name: study_boosters.best_params[f"w_{name}"] for name in booster_names}
    best_booster_norm = {name: best_booster_raw[name] / sum(best_booster_raw.values()) for name in booster_names}
    b_raw, b_tuned, b_mult, b_preds, _ = evaluate_blend(best_booster_norm)
    
    # 6. Detailed Comparison & Best Selection
    print("\n" + "=" * 65)
    print("FINAL ENSEMBLE COMPARISON SUMMARY")
    print("=" * 65)
    print(f"Previous Benchmark (Scenario_4_TreeBased):       Macro-F1 = 0.6797")
    print(f"Phase 17 Single Best (CatBoost solo):            Macro-F1 = {single_results['CatBoost']['tuned_f1']:.4f}")
    print(f"Phase 17 Boosters Only Ensemble:                 Macro-F1 = {b_tuned:.4f}")
    print(f"Phase 17 All 5 Models Optimized Ensemble:        Macro-F1 = {opt_tuned:.4f}")
    print("=" * 65)
    
    if opt_tuned >= b_tuned:
        final_best_name = "Phase17_All5_Optimized_Ensemble"
        final_best_f1 = opt_tuned
        final_weights = best_norm_w
        final_mult = opt_mult
        final_preds = opt_preds
    else:
        final_best_name = "Phase17_Boosters_Optimized_Ensemble"
        final_best_f1 = b_tuned
        final_weights = best_booster_norm
        final_mult = b_mult
        final_preds = b_preds
        
    print(f"\n🌟 Selected Champion Configuration: {final_best_name}")
    print(f"🌟 Champion Macro-F1: {final_best_f1:.4f} (Delta vs 0.6797: {final_best_f1 - 0.6797:+.4f})")
    print("\nOptimal Ensemble Weights:")
    for k, v in final_weights.items():
        print(f"  - {k:18s}: {v:.4f} ({v*100:.1f}%)")
    print(f"Optimal Threshold Multipliers: {final_mult.round(4).tolist()}")
    
    print("\nClassification Report:")
    target_names = ['Low', 'Medium', 'High']
    print(classification_report(y_true, final_preds, target_names=target_names, digits=4))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, final_preds))
    
    # 7. Log to MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    exp_name = "Phase17_Deep_Tuning_Ensemble"
    if not mlflow.get_experiment_by_name(exp_name):
        mlflow.create_experiment(exp_name)
    mlflow.set_experiment(exp_name)
    
    with mlflow.start_run(run_name=final_best_name):
        mlflow.log_metric("macro_f1_tuned", final_best_f1)
        mlflow.log_metric("delta_vs_p16_baseline", final_best_f1 - 0.6797)
        for k, v in final_weights.items():
            mlflow.log_param(f"weight_{k}", v)
        mlflow.log_params({
            "mult_low": final_mult[0],
            "mult_medium": final_mult[1],
            "mult_high": final_mult[2]
        })
        
    # Save Ensemble Configuration
    ensemble_config = {
        'champion_name': final_best_name,
        'macro_f1': final_best_f1,
        'weights': final_weights,
        'threshold_multipliers': final_mult.tolist(),
        'single_model_results': {k: {'raw_f1': v['raw_f1'], 'tuned_f1': v['tuned_f1']} for k, v in single_results.items()}
    }
    with open('experiments/phase17_ensemble_config.json', 'w') as f:
        json.dump(ensemble_config, f, indent=4)
    print("\n✅ Saved ensemble config to experiments/phase17_ensemble_config.json")
    
    # 8. Train on Full Dataset & Generate Test Submission
    print("\n" + "=" * 65)
    print("GENERATING TEST SUBMISSION FILE")
    print("=" * 65)
    df_test_raw = pd.read_csv('data/raw/test.csv')
    test_ids = df_test_raw['respondent_id']
    
    df_train_canon = apply_canonicalization(df_raw)
    df_test_canon = apply_canonicalization(df_test_raw)
    
    # Prepare Full Features for Test
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
    X_train_full_proc = prep_full.fit_transform(X_train_full)
    X_test_full_proc = prep_full.transform(X_test_full)
    
    # Prepare Winning Domain Features for Test
    print("Preparing Winning Domain features for full dataset & test set...")
    df_train_wd = add_winning_domain_features(df_train_full)
    df_test_wd = add_winning_domain_features(df_test_full)
    
    X_train_wd = df_train_wd.drop(columns=cols_to_drop, errors='ignore')
    X_test_wd = df_test_wd.drop(columns=[c for c in cols_to_drop if c != target_col], errors='ignore')
    for c in X_train_wd.columns:
        if c not in X_test_wd.columns:
            X_test_wd[c] = 0
    X_test_wd = X_test_wd[X_train_wd.columns]
    
    num_f_wd = X_train_wd.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    cat_f_wd = [c for c in X_train_wd.columns if c not in num_f_wd]
    prep_wd = build_preprocessor(num_f_wd, cat_f_wd)
    X_train_wd_proc = prep_wd.fit_transform(X_train_wd)
    X_test_wd_proc = prep_wd.transform(X_test_wd)
    
    sw_full_train = compute_xgb_sample_weights(y_true)
    
    test_predictions_dict = {}
    
    for name in final_weights:
        w = final_weights[name]
        if w < 1e-4:
            continue
        print(f"  Fitting {name} on 100% training data (weight = {w:.4f})...", flush=True)
        if name == 'CatBoost':
            m_params = phase17_params['CatBoost']['params'].copy()
            m_params.update({'auto_class_weights': 'Balanced', 'random_state': 42, 'thread_count': -1, 'verbose': False})
            m = CatBoostClassifier(**m_params)
            m.fit(X_train_full_proc, y_true)
            test_predictions_dict[name] = m.predict_proba(X_test_full_proc)
        elif name == 'XGBoost':
            m_params = phase17_params['XGBoost']['params'].copy()
            m_params.update({'random_state': 42, 'n_jobs': -1, 'eval_metric': 'mlogloss'})
            m = xgb.XGBClassifier(**m_params)
            m.fit(X_train_full_proc, y_true, sample_weight=sw_full_train)
            test_predictions_dict[name] = m.predict_proba(X_test_full_proc)
        elif name == 'LightGBM':
            m_params = phase17_params['LightGBM']['params'].copy()
            m_params.update({'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1, 'verbose': -1})
            m = lgb.LGBMClassifier(**m_params)
            m.fit(X_train_full_proc, y_true)
            test_predictions_dict[name] = m.predict_proba(X_test_full_proc)
        elif name == 'RandomForest_WD':
            m_params = phase17_params['RandomForest_WD']['params'].copy()
            m_params.update({'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1})
            m = RandomForestClassifier(**m_params)
            m.fit(X_train_wd_proc, y_true)
            test_predictions_dict[name] = m.predict_proba(X_test_wd_proc)
        elif name == 'ExtraTrees_WD':
            m_params = phase17_params['ExtraTrees_WD']['params'].copy()
            m_params.update({'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1})
            m = ExtraTreesClassifier(**m_params)
            m.fit(X_train_wd_proc, y_true)
            test_predictions_dict[name] = m.predict_proba(X_test_wd_proc)

    test_blended = np.zeros((len(test_ids), 3))
    total_active_w = sum(final_weights[name] for name in test_predictions_dict)
    for name, p in test_predictions_dict.items():
        test_blended += (final_weights[name] / total_active_w) * p
        
    adjusted_test = test_blended * final_mult
    test_hard_preds = np.argmax(adjusted_test, axis=1)
    test_label_preds = [inv_label_map[p] for p in test_hard_preds]
    
    sub_df = pd.DataFrame({
        'respondent_id': test_ids,
        'digital_academic_wellbeing_level': test_label_preds
    })
    
    sub_path = 'submission_phase17_deep_ensemble.csv'
    sub_df.to_csv(sub_path, index=False)
    print(f"\n🎉 Successfully created submission file: {sub_path}")
    print("Test Prediction Class Distribution:")
    print(sub_df['digital_academic_wellbeing_level'].value_counts(normalize=True).round(4) * 100)
    print(f"Total rows: {len(sub_df)}")
    print("=" * 65)


if __name__ == '__main__':
    main()
