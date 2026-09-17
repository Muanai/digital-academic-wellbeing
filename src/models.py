import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

def get_baseline_models(random_state: int = 42) -> dict:
    """
    Returns a dictionary of candidate baseline models with class_weight='balanced'.
    """
    models = {
        "LogisticRegression": LogisticRegression(
            class_weight='balanced', random_state=random_state, max_iter=1000
        ),
        "RandomForest": RandomForestClassifier(
            class_weight='balanced', random_state=random_state, n_estimators=100
        ),
        "LightGBM": lgb.LGBMClassifier(
            class_weight='balanced', random_state=random_state, n_estimators=100,
            verbose=-1
        ),
        # XGBoost handles class_weight via scale_pos_weight for binary, 
        # but for multiclass 'balanced' is not a direct parameter in the same way.
        # However, XGBClassifier does not have `class_weight='balanced'`.
        # For baseline, we will leave XGBoost without it or compute sample_weights later.
        # For now we'll just instantiate it.
        "XGBoost": xgb.XGBClassifier(
            random_state=random_state, n_estimators=100, use_label_encoder=False, 
            eval_metric='mlogloss'
        ),
        "CatBoost": CatBoostClassifier(
            auto_class_weights='Balanced', random_state=random_state, iterations=100,
            verbose=False
        )
    }
    return models

def build_baseline_pipeline(model, numeric_features: list, categorical_features: list) -> Pipeline:
    """
    Builds a central scikit-learn Pipeline with KNN Imputer for numerics
    and Simple Imputer + OneHotEncoder for categoricals.
    """
    # Numeric pipeline: KNNImputer -> StandardScaler
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
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
