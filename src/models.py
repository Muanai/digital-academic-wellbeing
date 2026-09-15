import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.base import BaseEstimator

def get_model(model_name: str, params: dict) -> BaseEstimator:
    """
    Factory function to get a model instance based on name and params.
    """
    model_name = model_name.lower()
    
    if model_name == "lightgbm":
        return lgb.LGBMRegressor(**params) if params.get('objective', '') == 'regression' else lgb.LGBMClassifier(**params)
    elif model_name == "xgboost":
        return xgb.XGBRegressor(**params) if params.get('objective', '') == 'regression' else xgb.XGBClassifier(**params)
    elif model_name == "catboost":
        return CatBoostRegressor(**params) if params.get('objective', '') == 'regression' else CatBoostClassifier(**params)
    else:
        raise ValueError(f"Model {model_name} not supported.")
