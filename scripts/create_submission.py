# %% [markdown]
# # Digital Academic Wellbeing - Kaggle Submission
# 
# This notebook is the final submission for the Kaggle Competition: **Digital Academic Wellbeing**.
# 
# ## Strategy
# - **Feature Engineering**: Extensive ratios, aggregates, binning, missingness flags, and peer group aggregates (Phase 3.5).
# - **Preprocessing**: Robust string parsing, dirty data cleaning (e.g. impossible screen time), `KNNImputer`, `QuantileTransformer`, and `OneHotEncoder`.
# - **Modeling**: 5-Fold Stratified CV on 5 tree-based models: CatBoost, XGBoost, LightGBM, RandomForest, and ExtraTrees.
# - **Ensembling**: Weighted soft-voting (optimized via Optuna in Phase 6).
# - **Decision Making**: Thresholds optimized using Nelder-Mead on OOF probabilities to maximize Macro-F1.

# %%
import numpy as np
import pandas as pd
import warnings
import re
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.model_selection import StratifiedKFold

# Models
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    import os
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything(42)

# %% [markdown]
# ## 1. Preprocessing & Feature Engineering
# (Self-contained implementations from the project's src modules)

# %%
# --- FROM src/preprocessing.py ---
def parse_age(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r'(\d+)')[0].astype(float)

def normalize_gender(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.strip()

def normalize_binary(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.strip()

def clean_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'notifications_per_day_clean' in df.columns:
        df.loc[df['notifications_per_day_clean'] < 0, 'notifications_per_day_clean'] = np.nan
    if 'screen_time_hrs_per_day_clean' in df.columns:
        impossible_mask = df['screen_time_hrs_per_day_clean'] > 24
        df.loc[impossible_mask, 'screen_time_hrs_per_day_clean'] = np.nan
    return df

def apply_canonicalization(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'age' in df.columns:
        df['age_clean'] = parse_age(df['age'])
    if 'gender' in df.columns:
        df['gender_clean'] = normalize_gender(df['gender'])
        
    binary_cols = ['uses_productivity_app', 'frequently_procrastinates', 
                   'family_academic_achievement_history', 'takes_extra_tutoring', 
                   'major_life_event_this_semester']
    for col in binary_cols:
        if col in df.columns:
            df[f'{col}_clean'] = normalize_binary(df[col])
            
    if 'notifications_per_day' in df.columns:
        df['notifications_per_day_clean'] = df['notifications_per_day'].astype(float)
    if 'screen_time_hrs_per_day' in df.columns:
        df['screen_time_hrs_per_day_clean'] = df['screen_time_hrs_per_day'].astype(float)
        
    df = clean_impossible_values(df)
    return df

# --- FROM src/features.py ---
def add_digital_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    eps = 1e-5
    df_feat['social_media_share'] = df_feat['social_media_hrs_per_day'] / (df_feat['screen_time_hrs_per_day'] + eps)
    df_feat['social_media_impact'] = df_feat['social_media_hrs_per_day'] * df_feat['screen_time_before_bed_minutes']
    return df_feat

def add_academic_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    eps = 1e-5
    df_feat['procrastination_stress_index'] = df_feat['procrastination_level_1_10'] * df_feat['academic_stress_level_1_10']
    df_feat['late_per_study_hour'] = df_feat['late_assignments_per_semester'] / (df_feat['study_hours_per_week'] + eps)
    return df_feat

def add_cross_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    eps = 1e-5
    df_feat['screen_to_sleep_ratio'] = (df_feat['screen_time_hrs_per_day'] / (df_feat['sleep_hours'] + eps)).clip(upper=50)
    df_feat['stress_gpa_ratio'] = (df_feat['academic_stress_level_1_10'] / (df_feat['gpa'] + eps)).clip(upper=50)
    df_feat['caffeine_to_sleep'] = (df_feat['caffeine_intake_mg_per_day'] / (df_feat['sleep_hours'] + eps)).clip(upper=200)
    df_feat['physical_vs_screen'] = (df_feat['physical_activity_hrs_per_week'] / (df_feat['screen_time_hrs_per_day'] + eps)).clip(upper=50)
    df_feat['gpa_x_attendance'] = df_feat['gpa'] * df_feat['class_attendance_percent']
    df_feat['study_vs_screen'] = (df_feat['study_hours_per_week'] / (df_feat['screen_time_hrs_per_day'] * 7 + eps)).clip(upper=50)
    df_feat['recovery_index'] = ((df_feat['sleep_hours'] * df_feat['diet_quality_1_10']) / (df_feat['caffeine_intake_mg_per_day'] + eps)).clip(upper=100)
    return df_feat

def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    df_feat['academic_pressure'] = (df_feat['academic_stress_level_1_10'] / 10) + (df_feat['procrastination_level_1_10'] / 10)
    df_feat['digital_overload'] = (df_feat['screen_time_hrs_per_day'] / 16) + (df_feat['social_media_hrs_per_day'] / 8)
    return df_feat

def add_binning_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    df_feat['is_sleep_deprived'] = (df_feat['sleep_hours'] < 6.0).astype(int)
    df_feat['is_highly_active'] = (df_feat['physical_activity_hrs_per_week'] >= 5.0).astype(int)
    df_feat['high_caffeine'] = (df_feat['caffeine_intake_mg_per_day'] > 250).astype(int)
    return df_feat

def add_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    missing_cols = ['caffeine_intake_mg_per_day', 'class_attendance_percent', 'social_media_hrs_per_day', 'study_hours_per_week', 'uses_productivity_app', 'favorite_content_type']
    cols_to_check = [col for col in missing_cols if col in df_feat.columns]
    if cols_to_check:
        df_feat['missing_count'] = df_feat[cols_to_check].isnull().sum(axis=1)
        for col in cols_to_check:
            df_feat[f'is_missing_{col}'] = df_feat[col].isnull().astype(int)
    return df_feat

def add_group_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    if 'field_of_study' in df_feat.columns:
        if 'gpa' in df_feat.columns:
            df_feat['gpa_field_mean'] = df_feat.groupby('field_of_study')['gpa'].transform('mean')
            df_feat['gpa_diff_field_mean'] = df_feat['gpa'] - df_feat['gpa_field_mean']
        if 'academic_stress_level_1_10' in df_feat.columns:
            df_feat['stress_field_mean'] = df_feat.groupby('field_of_study')['academic_stress_level_1_10'].transform('mean')
            df_feat['stress_diff_field_mean'] = df_feat['academic_stress_level_1_10'] - df_feat['stress_field_mean']
        if 'screen_time_hrs_per_day' in df_feat.columns:
            df_feat['screen_time_field_mean'] = df_feat.groupby('field_of_study')['screen_time_hrs_per_day'].transform('mean')
            df_feat['screen_time_diff_field_mean'] = df_feat['screen_time_hrs_per_day'] - df_feat['screen_time_field_mean']
    return df_feat

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    df_feat = add_digital_features(df_feat)
    df_feat = add_academic_features(df_feat)
    df_feat = add_cross_domain_features(df_feat)
    df_feat = add_aggregate_features(df_feat)
    df_feat = add_binning_features(df_feat)
    df_feat = add_missingness_features(df_feat)
    df_feat = add_group_aggregations(df_feat)
    return df_feat

# %% [markdown]
# ## 2. Data Loading

# %%
import os
train_path = 'train.csv' if os.path.exists('train.csv') else 'data/raw/train.csv'
test_path = 'test.csv' if os.path.exists('test.csv') else 'data/raw/test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Extract and clean Test IDs
test_ids = test_df['respondent_id']

# Concatenate for consistent group aggregations, then split back!
# Wait, group aggregation (transform) leaks test info into train if we concat?
# For a Kaggle notebook, test.csv is evaluated independently. But Kaggle test.csv usually comes all at once.
# In our Phase 3.6, we generated features on train and test separately, meaning test features were relative to test group means!
# Let's keep it separate to match Phase 3.6 exactly.

train_df = apply_canonicalization(train_df)
train_df = generate_features(train_df)

test_df = apply_canonicalization(test_df)
test_df = generate_features(test_df)

target_col = 'digital_academic_wellbeing_level'
label_map = {'Low': 0, 'Medium': 1, 'High': 2}
inv_label_map = {0: 'Low', 1: 'Medium', 2: 'High'}

y = train_df[target_col].map(label_map).values

# Identify feature columns
cols_to_drop_train = ['respondent_id', target_col]
# We also drop original raw columns if they got a `_clean` equivalent
raw_to_drop = [c.replace('_clean', '') for c in train_df.columns if c.endswith('_clean')]
cols_to_drop_train.extend(raw_to_drop)

X = train_df.drop(columns=cols_to_drop_train, errors='ignore')
X_test = test_df.drop(columns=['respondent_id'] + raw_to_drop, errors='ignore')

# Align columns (just in case)
missing_in_test = set(X.columns) - set(X_test.columns)
for col in missing_in_test:
    X_test[col] = 0
X_test = X_test[X.columns]

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Train shape: {X.shape}, Test shape: {X_test.shape}")

# %% [markdown]
# ## 3. Pipeline Configuration

# %%
def build_pipeline(model):
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
        ])
        
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

# Base models (using robust default hyperparameters as found in Phase 3.6 & 4)
models = {
    'CatBoost': CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, auto_class_weights='Balanced', random_seed=42, verbose=False),
    'XGBoost': XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42, eval_metric='mlogloss'), # Sample weights handled during fit
    'LightGBM': LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, class_weight='balanced', random_state=42, verbose=-1),
    'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
}

# XGBoost needs manual sample weights to simulate class_weight='balanced'
from sklearn.utils.class_weight import compute_sample_weight
xgb_sample_weight = compute_sample_weight(class_weight='balanced', y=y)

# Phase 6 Best Params
model_weights = {
    "CatBoost": 0.4351306370124723,
    "XGBoost": 0.4369588421825278,
    "LightGBM": 0.045755929481960285,
    "RandomForest": 0.029914118895709114,
    "ExtraTrees": 0.05224047242733051
}
threshold_multipliers = np.array([1.0904753550422448, 1.1287390471501415, 0.8611667292110078])

# %% [markdown]
# ## 4. Training (K-Fold Averaging)

# %%
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

test_probas_dict = {name: np.zeros((len(X_test), 3)) for name in models.keys()}
oof_probas_dict = {name: np.zeros((len(X), 3)) for name in models.keys()}

print("Starting 5-Fold Training...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"--- Fold {fold+1} ---")
    X_train_fold, y_train_fold = X.iloc[train_idx], y[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y[val_idx]
    
    for name, model in models.items():
        pipeline = build_pipeline(model)
        
        # Fit
        if name == 'XGBoost':
            sw_train = xgb_sample_weight[train_idx]
            pipeline.fit(X_train_fold, y_train_fold, classifier__sample_weight=sw_train)
        else:
            pipeline.fit(X_train_fold, y_train_fold)
            
        # OOF Predictions
        oof_probas_dict[name][val_idx] = pipeline.predict_proba(X_val_fold)
        
        # Test Predictions (accumulate for averaging)
        test_probas_dict[name] += pipeline.predict_proba(X_test) / skf.n_splits
        
print("Training Completed.")

# %% [markdown]
# ## 5. Ensembling & Threshold Application

# %%
print("\nApplying Ensemble Weights...")
final_test_proba = np.zeros((len(X_test), 3))
for name in models.keys():
    final_test_proba += test_probas_dict[name] * model_weights[name]
    
# Apply Optuna/Nelder-Mead threshold multipliers
adjusted_test_proba = final_test_proba * threshold_multipliers
final_test_preds = np.argmax(adjusted_test_proba, axis=1)

# Map back to original string labels
final_test_labels = [inv_label_map[p] for p in final_test_preds]

# %% [markdown]
# ## 6. Submission

# %%
submission = pd.DataFrame({
    'respondent_id': test_ids,
    'digital_academic_wellbeing_level': final_test_labels
})
submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")
print("\nClass Distribution:")
print(submission['digital_academic_wellbeing_level'].value_counts(normalize=True))
