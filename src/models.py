import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

def get_baseline_models(random_state: int = 42) -> dict:
    """
    Returns a dictionary of candidate baseline models with class_weight='balanced'.
    
    Phase 3.5 update:
    - ExtraTrees added for ensemble diversity (STRATEGY E004)
    - XGBoost note: use compute_sample_weights() helper at fit-time via evaluate_model
    """
    models = {
        "LogisticRegression": LogisticRegression(
            class_weight='balanced', random_state=random_state, max_iter=1000
        ),
        "RandomForest": RandomForestClassifier(
            class_weight='balanced', random_state=random_state, n_estimators=100
        ),
        "ExtraTrees": ExtraTreesClassifier(
            class_weight='balanced', random_state=random_state, n_estimators=100
        ),
        "MLPClassifier": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=1000, early_stopping=True, 
            random_state=random_state
        ),
        "LightGBM_s1": lgb.LGBMClassifier(
            class_weight='balanced', random_state=42, n_estimators=100, subsample=0.8, colsample_bytree=0.8, verbose=-1
        ),
        "LightGBM_s2": lgb.LGBMClassifier(
            class_weight='balanced', random_state=123, n_estimators=100, subsample=0.8, colsample_bytree=0.8, verbose=-1
        ),
        "LightGBM_s3": lgb.LGBMClassifier(
            class_weight='balanced', random_state=7, n_estimators=100, subsample=0.8, colsample_bytree=0.8, verbose=-1
        ),
        "XGBoost_s1": xgb.XGBClassifier(
            random_state=42, n_estimators=100, subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss'
        ),
        "XGBoost_s2": xgb.XGBClassifier(
            random_state=123, n_estimators=100, subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss'
        ),
        "XGBoost_s3": xgb.XGBClassifier(
            random_state=7, n_estimators=100, subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss'
        ),
        "CatBoost_s1": CatBoostClassifier(
            auto_class_weights='Balanced', random_state=42, iterations=100, verbose=False
        ),
        "CatBoost_s2": CatBoostClassifier(
            auto_class_weights='Balanced', random_state=123, iterations=100, verbose=False
        )
    }
    return models


def compute_xgb_sample_weights(y: pd.Series) -> np.ndarray:
    """
    Computes per-sample class-balanced weights for XGBoost.
    XGBoost lacks native class_weight='balanced'; this is the workaround.
    Uses sklearn's compute_sample_weight which mirrors the balanced formula.
    """
    return compute_sample_weight(class_weight='balanced', y=y)

def build_baseline_pipeline(model, numeric_features: list, categorical_features: list) -> Pipeline:
    """
    Builds a central scikit-learn Pipeline with KNN Imputer for numerics
    and Simple Imputer + OneHotEncoder for categoricals.
    """
    # Numeric pipeline: KNNImputer -> QuantileTransformer (robust heavy-tail normalization)
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', QuantileTransformer(output_distribution='normal', random_state=42))
    ])
    
    # Categorical pipeline: SimpleImputer (most frequent) -> OneHotEncoder
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    # Full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return pipeline
