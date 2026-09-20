import sys
import json
import warnings
import pandas as pd
import numpy as np
import optuna
import mlflow
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.preprocessing import apply_canonicalization
from src.features import get_model_specific_features
from src.validation import get_cv_strategy, evaluate_model, optimize_thresholds
from src.models import get_baseline_models, build_baseline_pipeline, get_regression_models
from src.utils import seed_everything
from scripts.exp_phase12_hyperparameter_tuning import precompute_folds

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

def load_best_params():
    try:
        with open('experiments/phase12_best_params.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def regressor_to_proba(preds):
    """Converts continuous regression predictions [0, 2] into pseudo-probabilities for 3 classes."""
    probas = np.zeros((len(preds), 3))
    preds_clipped = np.clip(preds, 0, 2)
    for i, p in enumerate(preds_clipped):
        d0 = abs(p - 0)
        d1 = abs(p - 1)
        d2 = abs(p - 2)
        w0 = 1 / (d0 + 1e-5)
        w1 = 1 / (d1 + 1e-5)
        w2 = 1 / (d2 + 1e-5)
        total = w0 + w1 + w2
        probas[i] = [w0/total, w1/total, w2/total]
    return probas

def evaluate_ensemble_variation(variation_name, model_names, oof_probas_dict, y_true):
    """Optimizes weights and thresholds for a specific combination of models."""
    print(f"\nEvaluating Variation: {variation_name} ({len(model_names)} models)")
    
    # Optuna weight optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        weights = []
        for name in model_names:
            weights.append(trial.suggest_float(f"weight_{name}", 0, 1))
        weights = np.array(weights)
        if weights.sum() == 0:
            return 0.0
        weights /= weights.sum()
        
        blended_proba = np.zeros((len(y_true), 3))
        for i, name in enumerate(model_names):
            blended_proba += oof_probas_dict[name] * weights[i]
            
        preds = np.argmax(blended_proba, axis=1)
        return f1_score(y_true, preds, average='macro')

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=150)
    
    best_weights = study.best_params
    total = sum(best_weights.values())
    norm_weights = {k.replace('weight_', ''): v / total for k, v in best_weights.items()}
    
    # Final blending
    ensemble_proba = np.zeros((len(y_true), 3))
    for name in model_names:
        ensemble_proba += oof_probas_dict[name] * norm_weights[name]
        
    # Nelder-Mead Optimization
    ensemble_f1, best_thresholds, _ = optimize_thresholds(pd.Series(y_true), ensemble_proba)
    
    print(f"  > Optimized Macro-F1: {ensemble_f1:.5f}")
    return ensemble_f1, norm_weights

def run_controlled_variations():
    seed_everything(42)
    print("Loading data...")
    df_raw = pd.read_csv('data/raw/train.csv')
    best_params = load_best_params()
    
    print("Applying canonicalization...")
    df = apply_canonicalization(df_raw)
    
    target_col = 'digital_academic_wellbeing_level'
    cols_to_drop = ['respondent_id', target_col]
    raw_to_drop = [c.replace('_clean', '') for c in df.columns if c.endswith('_clean')]
    cols_to_drop.extend(raw_to_drop)
    y = df[target_col].map({'Low': 0, 'Medium': 1, 'High': 2})
    len_df = len(df_raw)
    y_true_all = y.values
    
    cv = get_cv_strategy(n_splits=5, random_state=42)
    
    oof_probas_dict = {}
    
    # ==========================================
    # 1. Evaluate Golden Models (Phase 6 setup)
    # ==========================================
    print("\n--- 1. Evaluating Golden Models (Ablated Features) ---")
    golden_models = {
        'CatBoost_Golden': CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, auto_class_weights='Balanced', random_seed=42, verbose=False),
        'XGBoost_Golden': xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42, eval_metric='mlogloss'),
        'LightGBM_Golden': lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, class_weight='balanced', random_state=42, verbose=-1),
        'RandomForest_Golden': get_baseline_models(random_state=42)['RandomForest'],
        'ExtraTrees_Golden': get_baseline_models(random_state=42)['ExtraTrees']
    }
    
    for model_name, model in golden_models.items():
        print(f"  > Generating OOF for {model_name}...")
        model_type = "tree" if any(t in model_name for t in ["CatBoost", "XGBoost", "LightGBM", "RandomForest", "ExtraTrees"]) else "linear"
        df_feat = get_model_specific_features(df, model_type=model_type)
        X = df_feat.drop(columns=cols_to_drop, errors='ignore')
        
        numeric_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        categorical_features = [c for c in X.columns if c not in numeric_features]
        pipeline = build_baseline_pipeline(model, numeric_features, categorical_features)
        
        if "XGBoost" in model_name:
            from src.models import compute_xgb_sample_weights
            sw = compute_xgb_sample_weights(y)
            _, _, oof_proba = evaluate_model(pipeline, X, y, cv, sample_weight=sw)
        else:
            _, _, oof_proba = evaluate_model(pipeline, X, y, cv)
        oof_probas_dict[model_name] = oof_proba

    # ==========================================
    # 2. Evaluate Tuned & Regressor Models (Full Features)
    # ==========================================
    print("\n--- 2. Evaluating Tuned & Regressor Models (Full Features) ---")
    label_map = {'Low': 0, 'Medium': 1, 'High': 2}
    folds_data = precompute_folds(df_raw, target_col, label_map, cv)
    
    advanced_models = {
        'XGBoost_Regressor': get_regression_models(random_state=42)['XGBoost_Reg_s1']
    }
    
    if best_params:
        advanced_models.update({
            "LightGBM_Tuned": lgb.LGBMClassifier(**best_params['LightGBM']['params'], class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1),
            "XGBoost_Tuned": xgb.XGBClassifier(**best_params['XGBoost']['params'], random_state=42, eval_metric='mlogloss', n_jobs=-1),
            "CatBoost_Tuned": CatBoostClassifier(**best_params['CatBoost']['params'], auto_class_weights='Balanced', random_state=42, verbose=False, thread_count=-1)
        })
        
    for model_name, model in advanced_models.items():
        print(f"  > Generating OOF for {model_name}...")
        oof_proba = np.zeros((len_df, 3))
        oof_preds_reg = np.zeros(len_df)
        
        is_regressor = "Regressor" in model_name or "Reg" in model_name
        
        for fold_data in folds_data:
            if "XGBoost" in model_name and not is_regressor:
                model.fit(fold_data['X_train'], fold_data['y_train'], sample_weight=fold_data['sw_train'])
            elif "XGBoost" in model_name and is_regressor:
                # XGBRegressor supports sample_weight too, but let's just fit it normally for regression
                model.fit(fold_data['X_train'], fold_data['y_train'])
            else:
                model.fit(fold_data['X_train'], fold_data['y_train'])
                
            if is_regressor:
                preds = model.predict(fold_data['X_val'])
                oof_preds_reg[fold_data['val_idx']] = preds
            else:
                proba = model.predict_proba(fold_data['X_val'])
                oof_proba[fold_data['val_idx']] = proba
                
        if is_regressor:
            oof_proba = regressor_to_proba(oof_preds_reg)
            
        oof_probas_dict[model_name] = oof_proba

    # ==========================================
    # 3. Controlled Variations
    # ==========================================
    print("\n" + "="*50)
    print("PHASE 13: CONTROLLED VARIATIONS RESULTS")
    print("="*50)
    
    base_golden = ['CatBoost_Golden', 'XGBoost_Golden', 'LightGBM_Golden', 'RandomForest_Golden', 'ExtraTrees_Golden']
    
    variations = {
        "V0_Golden_Baseline": base_golden,
        "V1_Golden_+_XGB_Regressor": base_golden + ['XGBoost_Regressor'],
        "V2_Golden_Replace_XGB_w_Tuned": ['CatBoost_Golden', 'XGBoost_Tuned', 'LightGBM_Golden', 'RandomForest_Golden', 'ExtraTrees_Golden'],
        "V3_Golden_Replace_LGB_w_Tuned": ['CatBoost_Golden', 'XGBoost_Golden', 'LightGBM_Tuned', 'RandomForest_Golden', 'ExtraTrees_Golden'],
        "V4_Golden_Replace_CAT_w_Tuned": ['CatBoost_Tuned', 'XGBoost_Golden', 'LightGBM_Golden', 'RandomForest_Golden', 'ExtraTrees_Golden'],
        "V5_Golden_Replace_ALL_Trees_w_Tuned": ['CatBoost_Tuned', 'XGBoost_Tuned', 'LightGBM_Tuned', 'RandomForest_Golden', 'ExtraTrees_Golden']
    }
    
    results = []
    for var_name, models in variations.items():
        # Ensure we have all models for this variation
        if all(m in oof_probas_dict for m in models):
            f1, weights = evaluate_ensemble_variation(var_name, models, oof_probas_dict, y_true_all)
            results.append({'Variation': var_name, 'Macro_F1': f1})
        else:
            print(f"Skipping {var_name} due to missing models.")
            
    df_res = pd.DataFrame(results).sort_values('Macro_F1', ascending=False)
    print("\n" + "="*50)
    print("FINAL LEADERBOARD (CV MACRO-F1)")
    print("="*50)
    print(df_res.to_string(index=False))

if __name__ == '__main__':
    run_controlled_variations()
