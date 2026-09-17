import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────
# Feature Family Functions
# ─────────────────────────────────────────────────

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
    # Core cross-domain ratios (Phase 2 Error Analysis winners)
    df_feat['screen_to_sleep_ratio'] = df_feat['screen_time_hrs_per_day'] / (df_feat['sleep_hours'] + eps)
    df_feat['stress_gpa_ratio'] = df_feat['academic_stress_level_1_10'] / (df_feat['gpa'] + eps)
    df_feat['caffeine_to_sleep'] = df_feat['caffeine_intake_mg_per_day'] / (df_feat['sleep_hours'] + eps)
    df_feat['physical_vs_screen'] = df_feat['physical_activity_hrs_per_week'] / (df_feat['screen_time_hrs_per_day'] + eps)
    # Phase 3.5: Additional cross-domain features from STRATEGY.md
    df_feat['gpa_x_attendance'] = df_feat['gpa'] * df_feat['class_attendance_percent']
    df_feat['study_vs_screen'] = df_feat['study_hours_per_week'] / (df_feat['screen_time_hrs_per_day'] * 7 + eps)
    df_feat['recovery_index'] = (df_feat['sleep_hours'] * df_feat['diet_quality_1_10']) / (df_feat['caffeine_intake_mg_per_day'] + eps)
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

# ─────────────────────────────────────────────────
# Model-Specific Feature Selectors (for ablation)
# ─────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────
# Main Entry Point (used by train.py and scripts)
# ─────────────────────────────────────────────────

_FAMILY_MAP = {
    "digital": add_digital_features,
    "academic": add_academic_features,
    "cross_domain": add_cross_domain_features,
    "aggregate": add_aggregate_features,
    "binning": add_binning_features,
}

def generate_features(df: pd.DataFrame, families: list = None) -> pd.DataFrame:
    """
    Canonical entry point for feature engineering.
    
    Args:
        df:       Canonicalized DataFrame (output of apply_canonicalization).
        families: List of feature families to apply. If None, applies ALL families.
                  Valid values: 'digital', 'academic', 'cross_domain', 'aggregate', 'binning'.
    
    Returns:
        DataFrame with all requested engineered features added.
    """
    if families is None:
        families = list(_FAMILY_MAP.keys())
    
    df_feat = df.copy()
    for family in families:
        if family not in _FAMILY_MAP:
            raise ValueError(f"Unknown feature family: '{family}'. Valid: {list(_FAMILY_MAP.keys())}")
        df_feat = _FAMILY_MAP[family](df_feat)
    
    return df_feat
