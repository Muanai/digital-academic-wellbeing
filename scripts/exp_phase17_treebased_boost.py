"""
Phase 17: TreeBased Ensemble Boost
====================================
Strategies 2–5 to improve Scenario_4_TreeBased (current best: Macro-F1 = 0.6797).

Strategies:
  S2: Multi-Seed Diversity (CatBoost x3, XGBoost x3)
  S3: Feature-Diversified Sub-Ensembles (different feature sets per model family)
  S4: Probability Calibration (Isotonic regression on OOF probas)
  S5: Rank-Based Blending (rank percentiles instead of raw probabilities)

All experiments logged to MLflow experiment 'Phase17_TreeBased_Boost'.
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
import optuna
import mlflow
from sklearn.metrics import f1_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import rankdata
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.preprocessing import apply_canonicalization
from src.features import generate_features, add_winning_domain_features
from src.validation import get_cv_strategy, optimize_thresholds
from src.models import compute_xgb_sample_weights
from src.utils import seed_everything

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


# ─────────────────────────────────────────────────
# Data Loading & Preprocessing
# ─────────────────────────────────────────────────

def load_and_prepare():
    """Loads raw data, applies canonicalization and feature engineering."""
    df_raw = pd.read_csv('data/raw/train.csv')
    df = apply_canonicalization(df_raw)

    target_col = 'digital_academic_wellbeing_level'
    label_map = {'Low': 0, 'Medium': 1, 'High': 2}
    y = df[target_col].map(label_map)

    return df_raw, df, y, target_col, label_map


def prepare_features(df, target_col, feature_mode='full'):
    """
    Prepares feature matrix X for a given feature mode.

    feature_mode:
      'full'    -> generate_features() (all families)
      'winning' -> generate_features() + add_winning_domain_features()
    """
    if feature_mode == 'winning':
        df_feat = generate_features(df)
        df_feat = add_winning_domain_features(df_feat)
    else:
        df_feat = generate_features(df)

    cols_to_drop = ['respondent_id', target_col]
    raw_to_drop = [c.replace('_clean', '') for c in df_feat.columns if c.endswith('_clean')]
    cols_to_drop.extend(raw_to_drop)

    X = df_feat.drop(columns=cols_to_drop, errors='ignore')
    return X


def precompute_folds(df_raw, df, y, target_col, label_map, cv, feature_mode='full'):
    """Precomputes fold data arrays with preprocessing for fast model training."""
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import KNNImputer, SimpleImputer
    from sklearn.preprocessing import QuantileTransformer, OneHotEncoder

    folds_data = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(df_raw, df_raw[target_col])):
        df_train = apply_canonicalization(df_raw.iloc[train_idx].copy())
        df_val = apply_canonicalization(df_raw.iloc[val_idx].copy())

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

        # Align columns
        for c in X_train.columns:
            if c not in X_val.columns:
                X_val[c] = 0
        X_val = X_val[X_train.columns]

        numeric_features = X_train.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        categorical_features = [c for c in X_train.columns if c not in numeric_features]

        numeric_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', QuantileTransformer(output_distribution='normal', random_state=42))
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='drop')

        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)

        sw_train = compute_xgb_sample_weights(y_train)

        folds_data.append({
            'X_train': X_train_proc, 'y_train': y_train,
            'X_val': X_val_proc, 'y_val': y_val,
            'val_idx': val_idx, 'sw_train': sw_train,
        })

    return folds_data


# ─────────────────────────────────────────────────
# Model Definitions
# ─────────────────────────────────────────────────

def load_phase12_params():
    """Loads best hyperparameters from Phase 12."""
    try:
        with open('experiments/phase12_best_params.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def get_tree_models(seeds=None, use_tuned=True):
    """
    Returns tree-based models.

    If use_tuned=True, uses Phase 12 tuned params for CatBoost/XGBoost/LightGBM.
    seeds: dict mapping model family to list of seeds.
           Default: {'CatBoost': [42], 'XGBoost': [42]}
    """
    best_params = load_phase12_params() if use_tuned else None
    if seeds is None:
        seeds = {'CatBoost': [42], 'XGBoost': [42]}

    models = {}

    for seed in seeds.get('CatBoost', [42]):
        name = f"CatBoost_s{seed}"
        if best_params and 'CatBoost' in best_params:
            params = best_params['CatBoost']['params'].copy()
            params.update({'auto_class_weights': 'Balanced', 'random_state': seed, 'verbose': False, 'thread_count': -1})
        else:
            params = {'iterations': 1000, 'learning_rate': 0.03, 'depth': 6,
                      'auto_class_weights': 'Balanced', 'random_seed': seed, 'verbose': False}
        models[name] = ('CatBoost', CatBoostClassifier(**params))

    for seed in seeds.get('XGBoost', [42]):
        name = f"XGBoost_s{seed}"
        if best_params and 'XGBoost' in best_params:
            params = best_params['XGBoost']['params'].copy()
            params.update({'random_state': seed, 'eval_metric': 'mlogloss', 'n_jobs': -1})
        else:
            params = {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 5,
                      'random_state': seed, 'eval_metric': 'mlogloss'}
        models[name] = ('XGBoost', xgb.XGBClassifier(**params))

    # LightGBM (single seed)
    if best_params and 'LightGBM' in best_params:
        lgb_params = best_params['LightGBM']['params'].copy()
        lgb_params.update({'class_weight': 'balanced', 'random_state': 42, 'verbose': -1, 'n_jobs': -1})
    else:
        lgb_params = {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 5,
                      'class_weight': 'balanced', 'random_state': 42, 'verbose': -1}
    models['LightGBM'] = ('LightGBM', lgb.LGBMClassifier(**lgb_params))

    # RandomForest & ExtraTrees (stable defaults)
    models['RandomForest'] = ('RandomForest', RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1))
    models['ExtraTrees'] = ('ExtraTrees', ExtraTreesClassifier(
        n_estimators=300, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1))

    return models


# ─────────────────────────────────────────────────
# OOF Generation
# ─────────────────────────────────────────────────

def generate_oof(folds_data, model_family, model, len_df):
    """Generates OOF probabilities for a single model across precomputed folds."""
    oof_proba = np.zeros((len_df, 3))

    for fold_data in folds_data:
        if 'XGBoost' in model_family:
            model.fit(fold_data['X_train'], fold_data['y_train'], sample_weight=fold_data['sw_train'])
        else:
            model.fit(fold_data['X_train'], fold_data['y_train'])

        proba = model.predict_proba(fold_data['X_val'])
        oof_proba[fold_data['val_idx']] = proba

    return oof_proba


# ─────────────────────────────────────────────────
# Ensemble Optimization
# ─────────────────────────────────────────────────

def optimize_ensemble_weights(model_names, oof_probas_dict, y_true, n_trials=200):
    """Optimizes ensemble weights using Optuna + Nelder-Mead threshold optimization."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        weights = []
        for name in model_names:
            weights.append(trial.suggest_float(name, 0.0, 1.0))
        total = sum(weights)
        if total == 0:
            return 0.0
        weights = [w / total for w in weights]

        blended = np.zeros_like(oof_probas_dict[model_names[0]])
        for w, name in zip(weights, model_names):
            blended += w * oof_probas_dict[name]

        best_f1, _, _ = optimize_thresholds(pd.Series(y_true), blended)
        return best_f1

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    # Normalize best weights
    raw_weights = study.best_params
    total = sum(raw_weights.values())
    norm_weights = {k: v / total for k, v in raw_weights.items()}

    # Final blend with best weights
    final_proba = np.zeros_like(oof_probas_dict[model_names[0]])
    for name in model_names:
        final_proba += norm_weights[name] * oof_probas_dict[name]

    # Raw F1
    raw_preds = np.argmax(final_proba, axis=1)
    raw_f1 = f1_score(y_true, raw_preds, average='macro')

    # Optimized F1
    opt_f1, opt_multipliers, opt_preds = optimize_thresholds(pd.Series(y_true), final_proba)

    return {
        'raw_f1': raw_f1,
        'opt_f1': opt_f1,
        'weights': norm_weights,
        'multipliers': opt_multipliers,
        'preds': opt_preds,
        'proba': final_proba,
    }


# ─────────────────────────────────────────────────
# Strategy Implementations
# ─────────────────────────────────────────────────

def run_strategy_2(folds_data_full, y_true, len_df):
    """Strategy 2: Multi-Seed Diversity for CatBoost + XGBoost."""
    print("\n" + "=" * 60)
    print("STRATEGY 2: Multi-Seed Diversity")
    print("=" * 60)

    seeds = {'CatBoost': [42, 123, 7], 'XGBoost': [42, 123, 7]}
    models = get_tree_models(seeds=seeds, use_tuned=True)

    oof_dict = {}
    for name, (family, model) in models.items():
        print(f"  > Generating OOF for {name}...")
        oof_dict[name] = generate_oof(folds_data_full, family, model, len_df)

    model_names = list(oof_dict.keys())
    result = optimize_ensemble_weights(model_names, oof_dict, y_true, n_trials=200)

    print(f"\n  Raw Macro-F1: {result['raw_f1']:.4f}")
    print(f"  Optimized Macro-F1: {result['opt_f1']:.4f}")
    print(f"  Weights: {result['weights']}")
    print(f"\n{classification_report(y_true, result['preds'], target_names=['Low', 'Medium', 'High'])}")

    return result, oof_dict


def run_strategy_3(df_raw, df, y, target_col, label_map, cv, folds_data_full, len_df):
    """Strategy 3: Feature-Diversified Sub-Ensembles."""
    print("\n" + "=" * 60)
    print("STRATEGY 3: Feature-Diversified Sub-Ensembles")
    print("=" * 60)

    # Boosters: Full features (already precomputed in folds_data_full)
    booster_models = get_tree_models(seeds={'CatBoost': [42], 'XGBoost': [42]}, use_tuned=True)
    # Remove RF and ET from booster_models
    booster_models = {k: v for k, v in booster_models.items() if v[0] not in ('RandomForest', 'ExtraTrees')}

    oof_dict = {}
    for name, (family, model) in booster_models.items():
        print(f"  > Generating OOF for {name} (full features)...")
        oof_dict[name] = generate_oof(folds_data_full, family, model, len_df)

    # RF/ET: Winning domain features
    print("  > Precomputing winning-domain folds for RF/ET...")
    folds_data_winning = precompute_folds(df_raw, df, y, target_col, label_map, cv, feature_mode='winning')

    rf_et_models = {
        'RandomForest_WD': ('RandomForest', RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)),
        'ExtraTrees_WD': ('ExtraTrees', ExtraTreesClassifier(
            n_estimators=300, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)),
    }

    for name, (family, model) in rf_et_models.items():
        print(f"  > Generating OOF for {name} (winning-domain features)...")
        oof_dict[name] = generate_oof(folds_data_winning, family, model, len_df)

    model_names = list(oof_dict.keys())
    result = optimize_ensemble_weights(model_names, oof_dict, y.values, n_trials=200)

    print(f"\n  Raw Macro-F1: {result['raw_f1']:.4f}")
    print(f"  Optimized Macro-F1: {result['opt_f1']:.4f}")
    print(f"  Weights: {result['weights']}")
    print(f"\n{classification_report(y.values, result['preds'], target_names=['Low', 'Medium', 'High'])}")

    return result, oof_dict


def run_strategy_4(folds_data_full, y_true, len_df):
    """Strategy 4: Probability Calibration (Isotonic)."""
    print("\n" + "=" * 60)
    print("STRATEGY 4: Probability Calibration")
    print("=" * 60)

    models = get_tree_models(seeds={'CatBoost': [42], 'XGBoost': [42]}, use_tuned=True)

    oof_dict = {}
    for name, (family, model) in models.items():
        print(f"  > Generating calibrated OOF for {name}...")
        oof_proba = np.zeros((len_df, 3))

        for fold_data in folds_data_full:
            # Use inner 3-fold CV for calibration to prevent leakage
            calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)

            if 'XGBoost' in family:
                # CalibratedClassifierCV doesn't forward sample_weight cleanly for XGBoost,
                # so we fit the base model first then calibrate
                try:
                    calibrated_model.fit(fold_data['X_train'], fold_data['y_train'],
                                         sample_weight=fold_data['sw_train'])
                except TypeError:
                    # Fallback: fit without sample_weight for calibration wrapper
                    calibrated_model.fit(fold_data['X_train'], fold_data['y_train'])
            else:
                calibrated_model.fit(fold_data['X_train'], fold_data['y_train'])

            proba = calibrated_model.predict_proba(fold_data['X_val'])
            oof_proba[fold_data['val_idx']] = proba

            # Re-instantiate model for next fold (CalibratedClassifierCV modifies it)
            if family == 'CatBoost':
                model = CatBoostClassifier(**model.get_params())
            elif family == 'XGBoost':
                model = xgb.XGBClassifier(**model.get_params())
            elif family == 'LightGBM':
                model = lgb.LGBMClassifier(**model.get_params())
            elif family == 'RandomForest':
                model = RandomForestClassifier(**model.get_params())
            elif family == 'ExtraTrees':
                model = ExtraTreesClassifier(**model.get_params())

        oof_dict[name] = oof_proba

    model_names = list(oof_dict.keys())
    result = optimize_ensemble_weights(model_names, oof_dict, y_true, n_trials=200)

    print(f"\n  Raw Macro-F1: {result['raw_f1']:.4f}")
    print(f"  Optimized Macro-F1: {result['opt_f1']:.4f}")
    print(f"  Weights: {result['weights']}")
    print(f"\n{classification_report(y_true, result['preds'], target_names=['Low', 'Medium', 'High'])}")

    return result, oof_dict


def run_strategy_5(oof_probas_dict_baseline, y_true, model_names=None):
    """Strategy 5: Rank-Based Blending using baseline OOF probabilities."""
    print("\n" + "=" * 60)
    print("STRATEGY 5: Rank-Based Blending")
    print("=" * 60)

    if model_names is None:
        model_names = list(oof_probas_dict_baseline.keys())

    # Convert raw probabilities to rank percentiles
    ranked_dict = {}
    for name in model_names:
        proba = oof_probas_dict_baseline[name]
        ranked = np.zeros_like(proba)
        for cls in range(proba.shape[1]):
            ranks = rankdata(proba[:, cls])
            ranked[:, cls] = ranks / len(ranks)  # Normalize to [0, 1]
        ranked_dict[name] = ranked

    result = optimize_ensemble_weights(model_names, ranked_dict, y_true, n_trials=200)

    print(f"\n  Raw Macro-F1: {result['raw_f1']:.4f}")
    print(f"  Optimized Macro-F1: {result['opt_f1']:.4f}")
    print(f"  Weights: {result['weights']}")
    print(f"\n{classification_report(y_true, result['preds'], target_names=['Low', 'Medium', 'High'])}")

    return result


# ─────────────────────────────────────────────────
# MLflow Logging
# ─────────────────────────────────────────────────

def log_strategy_to_mlflow(strategy_name, result, y_true):
    """Logs a strategy's results to MLflow."""
    with mlflow.start_run(run_name=strategy_name):
        mlflow.log_param("strategy", strategy_name)

        # Metrics
        mlflow.log_metric("oof_macro_f1_raw", result['raw_f1'])
        mlflow.log_metric("oof_macro_f1_optimized", result['opt_f1'])

        # Per-class F1
        report = classification_report(y_true, result['preds'],
                                       target_names=['Low', 'Medium', 'High'], output_dict=True)
        mlflow.log_metric("f1_low", report['Low']['f1-score'])
        mlflow.log_metric("f1_medium", report['Medium']['f1-score'])
        mlflow.log_metric("f1_high", report['High']['f1-score'])

        # Weights
        if 'weights' in result:
            for k, v in result['weights'].items():
                mlflow.log_param(f"weight_{k}", round(v, 6))

        # Multipliers
        if 'multipliers' in result and result['multipliers'] is not None:
            mlflow.log_param("threshold_multipliers", str(list(np.round(result['multipliers'], 6))))


# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────

def main():
    seed_everything(42)

    print("=" * 60)
    print("PHASE 17: TreeBased Ensemble Boost")
    print("Strategies 2–5")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading and preparing data...")
    df_raw, df, y, target_col, label_map = load_and_prepare()
    len_df = len(df_raw)
    y_values = y.values
    cv = get_cv_strategy(n_splits=5, random_state=42)

    # 2. Precompute folds (full features)
    print("\n[2/6] Precomputing fold arrays (full features)...")
    folds_data_full = precompute_folds(df_raw, df, y, target_col, label_map, cv, feature_mode='full')

    # 3. Setup MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = "Phase17_TreeBased_Boost"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # ─── Baseline: Reproduce Scenario_4_TreeBased ───
    print("\n[3/6] Generating baseline OOF (Scenario_4_TreeBased)...")
    baseline_models = get_tree_models(seeds={'CatBoost': [42], 'XGBoost': [42]}, use_tuned=True)
    baseline_oof = {}
    for name, (family, model) in baseline_models.items():
        print(f"  > {name}...")
        baseline_oof[name] = generate_oof(folds_data_full, family, model, len_df)

    baseline_result = optimize_ensemble_weights(
        list(baseline_oof.keys()), baseline_oof, y_values, n_trials=200)
    print(f"\n  Baseline Macro-F1: {baseline_result['opt_f1']:.4f}")
    log_strategy_to_mlflow("S0_Baseline_TreeBased", baseline_result, y_values)

    # ─── Strategy 2: Multi-Seed Diversity ───
    print("\n[4/6] Running Strategy 2: Multi-Seed Diversity...")
    s2_result, s2_oof = run_strategy_2(folds_data_full, y_values, len_df)
    log_strategy_to_mlflow("S2_MultiSeed_Diversity", s2_result, y_values)

    # ─── Strategy 3: Feature-Diversified ───
    print("\n[5/6] Running Strategy 3: Feature-Diversified Sub-Ensembles...")
    s3_result, s3_oof = run_strategy_3(df_raw, df, y, target_col, label_map, cv, folds_data_full, len_df)
    log_strategy_to_mlflow("S3_Feature_Diversified", s3_result, y.values)

    # ─── Strategy 4: Probability Calibration ───
    print("\n[6/6] Running Strategy 4: Probability Calibration...")
    s4_result, s4_oof = run_strategy_4(folds_data_full, y_values, len_df)
    log_strategy_to_mlflow("S4_Probability_Calibration", s4_result, y_values)

    # ─── Strategy 5: Rank-Based Blending (using baseline OOF) ───
    print("\n[BONUS] Running Strategy 5: Rank-Based Blending...")
    s5_result = run_strategy_5(baseline_oof, y_values)
    log_strategy_to_mlflow("S5_Rank_Based_Blending", s5_result, y_values)

    # ─── Strategy 5b: Rank-Based on Multi-Seed OOF ───
    print("\n[BONUS] Running Strategy 5b: Rank-Based on Multi-Seed OOF...")
    s5b_result = run_strategy_5(s2_oof, y_values)
    log_strategy_to_mlflow("S5b_Rank_MultiSeed", s5b_result, y_values)

    # ─────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 17 RESULTS SUMMARY")
    print("=" * 60)

    all_results = {
        "S0_Baseline": baseline_result['opt_f1'],
        "S2_MultiSeed": s2_result['opt_f1'],
        "S3_FeatureDiversified": s3_result['opt_f1'],
        "S4_Calibration": s4_result['opt_f1'],
        "S5_RankBlend": s5_result['opt_f1'],
        "S5b_RankMultiSeed": s5b_result['opt_f1'],
    }

    for name, f1 in sorted(all_results.items(), key=lambda x: x[1], reverse=True):
        delta = f1 - baseline_result['opt_f1']
        marker = "**" if f1 == max(all_results.values()) else "  "
        print(f"  {marker} {name:30s} Macro-F1: {f1:.4f}  (delta {delta:+.4f})")

    best_name = max(all_results, key=all_results.get)
    print(f"\n  BEST STRATEGY: {best_name} -> Macro-F1: {all_results[best_name]:.4f}")
    print(f"  vs Scenario_4_TreeBased baseline: {baseline_result['opt_f1']:.4f}")


if __name__ == '__main__':
    main()
