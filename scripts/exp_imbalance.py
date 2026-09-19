import sys
import pandas as pd
import numpy as np
import mlflow

sys.path.insert(0, '.')
from src.preprocessing import apply_canonicalization
from src.features import generate_features
from src.validation import get_cv_strategy, evaluate_model, optimize_thresholds
from src.models import get_baseline_models, compute_xgb_sample_weights
from src.utils import seed_everything

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def build_imb_pipeline(model, numeric_features: list, categorical_features: list) -> ImbPipeline:
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
    
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    
    return pipeline

def run_experiment():
    seed_everything(42)
    print("Loading data...")
    df_raw = pd.read_csv('data/raw/train.csv')
    
    print("Applying base preprocessing and features...")
    df = apply_canonicalization(df_raw)
    df = generate_features(df)
    
    # Prepare X, y
    target_col = 'digital_academic_wellbeing_level'
    cols_to_drop = ['respondent_id', target_col]
    raw_to_drop = [c.replace('_clean', '') for c in df.columns if c.endswith('_clean')]
    cols_to_drop.extend(raw_to_drop)
    
    X = df.drop(columns=cols_to_drop, errors='ignore')
    y = df[target_col].map({'Low': 0, 'Medium': 1, 'High': 2})
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    cv = get_cv_strategy(n_splits=5, random_state=42)
    models = get_baseline_models(random_state=42)
    xgb_sample_weight = compute_xgb_sample_weights(y)
    
    mlflow.set_experiment("Phase8_Advanced")
    with mlflow.start_run(run_name="Exp_E_SMOTE"):
        mlflow.log_param("experiment", "Exp_E_SMOTE")
        
        results = []
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            # We use build_imb_pipeline to include SMOTE
            pipeline = build_imb_pipeline(model, numeric_features, categorical_features)
            
            # NOTE: XGBoost sample_weight might clash with SMOTE, we can pass it, but SMOTE changes row counts.
            # wait, if SMOTE changes the number of rows in X_train, passing sample_weight directly 
            # to fit() will raise a ValueError because len(sample_weight) != len(X_train_resampled).
            # We should probably NOT use xgb_sample_weight when SMOTE is active, since SMOTE itself balances the classes!
            # So we set sw to None for all models when using SMOTE.
            sw = None 
            
            # CatBoost handles class_weight='balanced' natively. To ensure fair testing, we should let SMOTE do the balancing.
            # But the baseline models are initialized with class_weight='balanced'. We will just let them run.
            
            overall_score, oof_preds, oof_proba = evaluate_model(pipeline, X, y, cv, sample_weight=sw)
            
            best_f1, best_weights, best_preds = optimize_thresholds(y, oof_proba)
            final_score = best_f1 if best_f1 is not None else overall_score
            
            results.append({'Model': model_name, 'Macro-F1 (Tuned)': final_score})
            mlflow.log_metric(f"{model_name}_macro_f1", final_score)
            print(f"{model_name} Score: {final_score:.4f}")
            
        res_df = pd.DataFrame(results).sort_values('Macro-F1 (Tuned)', ascending=False)
        print("\n=== EXP E: SMOTE IMBALANCE HANDLING RESULTS ===")
        print(res_df.to_string(index=False))

if __name__ == '__main__':
    run_experiment()
