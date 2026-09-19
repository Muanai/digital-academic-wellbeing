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
    df_feat['screen_to_sleep_ratio'] = (df_feat['screen_time_hrs_per_day'] / (df_feat['sleep_hours'] + eps)).clip(upper=50)
    df_feat['stress_gpa_ratio'] = (df_feat['academic_stress_level_1_10'] / (df_feat['gpa'] + eps)).clip(upper=50)
    df_feat['caffeine_to_sleep'] = (df_feat['caffeine_intake_mg_per_day'] / (df_feat['sleep_hours'] + eps)).clip(upper=200)
    df_feat['physical_vs_screen'] = (df_feat['physical_activity_hrs_per_week'] / (df_feat['screen_time_hrs_per_day'] + eps)).clip(upper=50)
    # Phase 3.5: Additional cross-domain features from STRATEGY.md
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
    # Define columns known to have missingness from Phase 1 audit
    missing_cols = [
        'caffeine_intake_mg_per_day', 'class_attendance_percent', 
        'social_media_hrs_per_day', 'study_hours_per_week', 
        'uses_productivity_app', 'favorite_content_type'
    ]
    
    # Only process columns that actually exist in the current dataframe
    cols_to_check = [col for col in missing_cols if col in df_feat.columns]
    
    if cols_to_check:
        df_feat['missing_count'] = df_feat[cols_to_check].isnull().sum(axis=1)
        for col in cols_to_check:
            df_feat[f'is_missing_{col}'] = df_feat[col].isnull().astype(int)
            
    return df_feat

def add_group_aggregations(df: pd.DataFrame, stats_df: pd.DataFrame = None) -> pd.DataFrame:
    df_feat = df.copy()
    eps = 1e-5
    
    # If stats_df is not provided, we compute stats on the dataframe itself (e.g. for training set)
    if stats_df is None:
        stats_df = df
        
    target_num_cols = ['gpa', 'academic_stress_level_1_10', 'study_hours_per_week',
                       'sleep_hours', 'screen_time_hrs_per_day', 'procrastination_level_1_10']
    
    group_keys = ['field_of_study', 'uses_productivity_app_clean', 'gender_clean']
    
    for grp in group_keys:
        if grp in df_feat.columns and grp in stats_df.columns:
            for num in target_num_cols:
                if num in df_feat.columns and num in stats_df.columns:
                    mean_col = f'{num}_by_{grp}_mean'
                    std_col = f'{num}_by_{grp}_std'
                    
                    # Calculate stats from stats_df
                    grp_stats = stats_df.groupby(grp)[num].agg(['mean', 'std']).reset_index()
                    grp_stats.columns = [grp, mean_col, std_col]
                    
                    # Merge into df_feat
                    df_feat = df_feat.merge(grp_stats, on=grp, how='left')
                    
                    # Difference and Z-Score
                    df_feat[f'{num}_diff_{grp}'] = df_feat[num] - df_feat[mean_col]
                    df_feat[f'{num}_z_{grp}'] = df_feat[f'{num}_diff_{grp}'] / (df_feat[std_col] + eps)
            
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

def add_winning_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    eps = 1e-5
    
    if all(c in df_feat.columns for c in ['sleep_hours', 'gpa', 'screen_time_hrs_per_day', 'social_media_hrs_per_day', 'class_attendance_percent', 'study_hours_per_week', 'procrastination_level_1_10', 'academic_stress_level_1_10', 'pre_exam_anxiety_level_1_10', 'screen_time_before_bed_minutes']):
        sleep = df_feat['sleep_hours']
        gpa = df_feat['gpa']
        screen = df_feat['screen_time_hrs_per_day']
        soc = df_feat['social_media_hrs_per_day']
        att = df_feat['class_attendance_percent']
        study = df_feat['study_hours_per_week']
        proc = df_feat['procrastination_level_1_10']
        stress = df_feat['academic_stress_level_1_10']
        anx = df_feat['pre_exam_anxiety_level_1_10']
        bed_scr = df_feat['screen_time_before_bed_minutes']
        
        # A. Wellbeing Ratios
        pos = (sleep / 7.5) * (gpa / 2.5) * (att / 80.0)
        neg = (proc / 5.0) * (bed_scr / 30.0 + 0.1) * (soc / 4.0 + 0.1) * (stress / 5.0)
        df_feat['wellbeing_force_ratio'] = pos / (neg + 0.05)
        df_feat['academic_health'] = gpa * (att / 100.0)
        df_feat['academic_strain'] = (stress + anx + proc) / (gpa + 0.1)
        
        # B. Sleep Interactions
        df_feat['sleep_deficit'] = sleep - 7.5
        df_feat['sleep_efficiency'] = sleep / (sleep + (bed_scr / 60.0) + 0.1)
        
        # C. Digital Burden
        df_feat['digital_burden'] = (soc + (bed_scr / 60.0)) / (sleep + 0.1)
        df_feat['digital_intensity'] = (screen + soc + (bed_scr / 60.0)) / (sleep + study + 0.1)
        
        # D. Psychology Products
        df_feat['composite_stress'] = (stress + anx + proc) / 3.0
        df_feat['stress_anxiety_product'] = stress * anx

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
    "missingness": add_missingness_features,
    "winning_domain": add_winning_domain_features,
    "group_aggregations": add_group_aggregations,
}

def generate_features(df: pd.DataFrame, families: list = None, stats_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Canonical entry point for feature engineering.
    
    Args:
        df:       Canonicalized DataFrame (output of apply_canonicalization).
        families: List of feature families to apply. If None, applies ALL families.
        stats_df: Optional DataFrame to use for calculating group statistics (prevents leakage).
                  If None, stats are calculated on `df` itself.
    
    Returns:
        DataFrame with all requested engineered features added.
    """
    if families is None:
        families = list(_FAMILY_MAP.keys())
    
    df_feat = df.copy()
    for family in families:
        if family not in _FAMILY_MAP:
            raise ValueError(f"Unknown feature family: '{family}'. Valid: {list(_FAMILY_MAP.keys())}")
        
        if family == "group_aggregations":
            df_feat = _FAMILY_MAP[family](df_feat, stats_df=stats_df)
        else:
            df_feat = _FAMILY_MAP[family](df_feat)
    
    return df_feat
