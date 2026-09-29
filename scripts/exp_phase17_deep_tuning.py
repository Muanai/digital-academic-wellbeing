import sys
import json
import warnings
import pandas as pd
import numpy as np
import optuna
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

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

def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', QuantileTransformer(output_distribution='normal', random_state=42))
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor

def precompute_folds(df_raw, target_col, label_map, cv, feature_mode='full'):
    print(f"Precomputing fold features ({feature_mode}) & preprocessed arrays to accelerate Optuna trials...", flush=True)
    folds_data = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(df_raw, df_raw[target_col])):
        print(f"  Processing Fold {fold+1}/5...", flush=True)
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
        
    print("Precomputation finished! All arrays are in memory ready for fast model training.\n", flush=True)
    return folds_data

def objective_lgb(trial, folds_data, len_df):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 80, 600),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    oof_proba = np.zeros((len_df, 3))
    y_true_all = np.zeros(len_df)
    
    for fold_data in folds_data:
        model = lgb.LGBMClassifier(**params)
        model.fit(fold_data['X_train'], fold_data['y_train'])
        
        proba = model.predict_proba(fold_data['X_val'])
        oof_proba[fold_data['val_idx']] = proba
        y_true_all[fold_data['val_idx']] = fold_data['y_val']
        
    best_f1, _, _ = optimize_thresholds(pd.Series(y_true_all), oof_proba)
    if best_f1 is None:
        best_f1 = f1_score(y_true_all, np.argmax(oof_proba, axis=1), average='macro')
    return best_f1

def objective_xgb(trial, folds_data, len_df):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 80, 600),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'mlogloss'
    }
    
    oof_proba = np.zeros((len_df, 3))
    y_true_all = np.zeros(len_df)
    
    for fold_data in folds_data:
        model = xgb.XGBClassifier(**params)
        model.fit(fold_data['X_train'], fold_data['y_train'], sample_weight=fold_data['sw_train'])
        
        proba = model.predict_proba(fold_data['X_val'])
        oof_proba[fold_data['val_idx']] = proba
        y_true_all[fold_data['val_idx']] = fold_data['y_val']
        
    best_f1, _, _ = optimize_thresholds(pd.Series(y_true_all), oof_proba)
    if best_f1 is None:
        best_f1 = f1_score(y_true_all, np.argmax(oof_proba, axis=1), average='macro')
    return best_f1

def objective_cat(trial, folds_data, len_df):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 600),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 20.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-4, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'auto_class_weights': 'Balanced',
        'random_state': 42,
        'thread_count': -1,
        'verbose': False
    }
    
    oof_proba = np.zeros((len_df, 3))
    y_true_all = np.zeros(len_df)
    
    for fold_data in folds_data:
        model = CatBoostClassifier(**params)
        model.fit(fold_data['X_train'], fold_data['y_train'])
        
        proba = model.predict_proba(fold_data['X_val'])
        oof_proba[fold_data['val_idx']] = proba
        y_true_all[fold_data['val_idx']] = fold_data['y_val']
        
    best_f1, _, _ = optimize_thresholds(pd.Series(y_true_all), oof_proba)
    if best_f1 is None:
        best_f1 = f1_score(y_true_all, np.argmax(oof_proba, axis=1), average='macro')
    return best_f1

def objective_rf(trial, folds_data, len_df):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 5, 25),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    
    oof_proba = np.zeros((len_df, 3))
    y_true_all = np.zeros(len_df)
    
    for fold_data in folds_data:
        model = RandomForestClassifier(**params)
        model.fit(fold_data['X_train'], fold_data['y_train'])
        
        proba = model.predict_proba(fold_data['X_val'])
        oof_proba[fold_data['val_idx']] = proba
        y_true_all[fold_data['val_idx']] = fold_data['y_val']
        
    best_f1, _, _ = optimize_thresholds(pd.Series(y_true_all), oof_proba)
    if best_f1 is None:
        best_f1 = f1_score(y_true_all, np.argmax(oof_proba, axis=1), average='macro')
    return best_f1

def objective_et(trial, folds_data, len_df):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 5, 25),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    
    oof_proba = np.zeros((len_df, 3))
    y_true_all = np.zeros(len_df)
    
    for fold_data in folds_data:
        model = ExtraTreesClassifier(**params)
        model.fit(fold_data['X_train'], fold_data['y_train'])
        
        proba = model.predict_proba(fold_data['X_val'])
        oof_proba[fold_data['val_idx']] = proba
        y_true_all[fold_data['val_idx']] = fold_data['y_val']
        
    best_f1, _, _ = optimize_thresholds(pd.Series(y_true_all), oof_proba)
    if best_f1 is None:
        best_f1 = f1_score(y_true_all, np.argmax(oof_proba, axis=1), average='macro')
    return best_f1

def run_tuning():
    seed_everything(42)
    print("Loading raw data...", flush=True)
    df_raw = pd.read_csv('data/raw/train.csv')
    len_df = len(df_raw)
    
    cv = get_cv_strategy(n_splits=5, random_state=42)
    target_col = 'digital_academic_wellbeing_level'
    label_map = {'Low': 0, 'Medium': 1, 'High': 2}
    
    # Booster models use full features
    print("\n--- Precomputing for Boosters (Full Features) ---", flush=True)
    folds_data_full = precompute_folds(df_raw, target_col, label_map, cv, feature_mode='full')
    
    # RF/ET use winning domain features (ablation) based on S3 Phase 17 findings
    print("\n--- Precomputing for RF/ET (Winning Domain Features) ---", flush=True)
    folds_data_winning = precompute_folds(df_raw, target_col, label_map, cv, feature_mode='winning')
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Phase17_Deep_Tuning")
    
    best_params_all = {}
    
    # model_name : (objective_func, folds_data_to_use)
    models_to_tune = {
        'LightGBM': (objective_lgb, folds_data_full),
        'XGBoost': (objective_xgb, folds_data_full),
        'CatBoost': (objective_cat, folds_data_full),
        'RandomForest_WD': (objective_rf, folds_data_winning),
        'ExtraTrees_WD': (objective_et, folds_data_winning)
    }
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    N_TRIALS = 100
    
    # Load existing best params if resuming
    import os
    if os.path.exists('experiments/phase17_best_params.json'):
        with open('experiments/phase17_best_params.json', 'r') as f:
            best_params_all = json.load(f)
    else:
        best_params_all = {}
    
    for model_name, (objective_func, folds_data) in models_to_tune.items():
        if model_name in best_params_all:
            print(f"Skipping {model_name} (already found in experiments/phase17_best_params.json)", flush=True)
            continue

        print(f"==========================================", flush=True)
        print(f"Starting Deep Tuning for {model_name} ({N_TRIALS} trials)", flush=True)
        print(f"==========================================", flush=True)
        
        with mlflow.start_run(run_name=f"DeepTuning_{model_name}"):
            def callback(study, trial):
                print(f"  [{model_name}] Trial {trial.number+1:03d}/{N_TRIALS} | Current F1: {trial.value:.4f} | Best so far: {study.best_value:.4f}", flush=True)
                
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(lambda trial: objective_func(trial, folds_data, len_df), n_trials=N_TRIALS, callbacks=[callback])
            
            print(f"\n>>> Best {model_name} Macro-F1: {study.best_value:.4f}", flush=True)
            print(f">>> Best Params: {study.best_params}\n", flush=True)
            
            mlflow.log_metric("best_macro_f1", study.best_value)
            mlflow.log_params(study.best_params)
            
            best_params_all[model_name] = {
                'best_macro_f1': study.best_value,
                'params': study.best_params
            }
            # Save immediately after each model finishes
            with open('experiments/phase17_best_params.json', 'w') as f:
                json.dump(best_params_all, f, indent=4)
        
    print("==========================================", flush=True)
    print("Phase 17 Deep Tuning Complete! Best parameters saved to experiments/phase17_best_params.json", flush=True)
    print("Logged to sqlite:///mlflow.db under experiment 'Phase17_Deep_Tuning'", flush=True)
    print("==========================================", flush=True)

if __name__ == '__main__':
    run_tuning()
