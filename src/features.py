import pandas as pd
import numpy as np

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
    df_feat['screen_to_sleep_ratio'] = df_feat['screen_time_hrs_per_day'] / (df_feat['sleep_hours'] + eps)
    df_feat['stress_gpa_ratio'] = df_feat['academic_stress_level_1_10'] / (df_feat['gpa'] + eps)
    df_feat['caffeine_to_sleep'] = df_feat['caffeine_intake_mg_per_day'] / (df_feat['sleep_hours'] + eps)
    df_feat['physical_vs_screen'] = df_feat['physical_activity_hrs_per_week'] / (df_feat['screen_time_hrs_per_day'] + eps)
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

def get_model_specific_features(df: pd.DataFrame, model_type: str = "tree") -> pd.DataFrame:
    """
    Returns the specific feature set optimized for either tree-based or linear models.
    """
    df_feat = df.copy()
    eps = 1e-5
    
    if model_type == "tree":
        # Tree models thrive on the core interaction features from our Phase 2 Error Analysis
        # (This is the exact combination that yielded the 0.6453 Macro-F1 on XGBoost)
        df_feat['screen_to_sleep_ratio'] = df_feat['screen_time_hrs_per_day'] / (df_feat['sleep_hours'] + eps)
        df_feat['social_media_impact'] = df_feat['social_media_hrs_per_day'] * df_feat['screen_time_before_bed_minutes']
        df_feat['stress_gpa_ratio'] = df_feat['academic_stress_level_1_10'] / (df_feat['gpa'] + eps)
        df_feat['procrastination_stress_index'] = df_feat['procrastination_level_1_10'] * df_feat['academic_stress_level_1_10']
        df_feat['caffeine_to_sleep'] = df_feat['caffeine_intake_mg_per_day'] / (df_feat['sleep_hours'] + eps)
        df_feat['physical_vs_screen'] = df_feat['physical_activity_hrs_per_week'] / (df_feat['screen_time_hrs_per_day'] + eps)
        
    elif model_type == "linear":
        # Linear models benefit from binned thresholds and linear aggregate scores
        df_feat['is_sleep_deprived'] = (df_feat['sleep_hours'] < 6.0).astype(int)
        df_feat['is_highly_active'] = (df_feat['physical_activity_hrs_per_week'] >= 5.0).astype(int)
        df_feat['high_caffeine'] = (df_feat['caffeine_intake_mg_per_day'] > 250).astype(int)
        df_feat['academic_pressure'] = (df_feat['academic_stress_level_1_10'] / 10) + (df_feat['procrastination_level_1_10'] / 10)
        df_feat['digital_overload'] = (df_feat['screen_time_hrs_per_day'] / 16) + (df_feat['social_media_hrs_per_day'] / 8)
        
    return df_feat
