import pandas as pd
import numpy as np

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate domain-specific features for Digital Academic Wellbeing based on Phase 2 error analysis.
    """
    df_feat = df.copy()
    
    # 1. Digital-to-Sleep Efficiency
    # To avoid division by zero, add a small epsilon (1e-5)
    eps = 1e-5
    df_feat['screen_to_sleep_ratio'] = df_feat['screen_time_hrs_per_day'] / (df_feat['sleep_hours'] + eps)
    df_feat['social_media_impact'] = df_feat['social_media_hrs_per_day'] * df_feat['screen_time_before_bed_minutes']
    
    # 2. Academic Stress Buffering
    # How well they cope: high stress but high GPA is a good sign
    df_feat['stress_gpa_ratio'] = df_feat['academic_stress_level_1_10'] / (df_feat['gpa'] + eps)
    df_feat['procrastination_stress_index'] = df_feat['procrastination_level_1_10'] * df_feat['academic_stress_level_1_10']
    
    # 3. Lifestyle Balance
    # Caffeine relative to sleep hours
    df_feat['caffeine_to_sleep'] = df_feat['caffeine_intake_mg_per_day'] / (df_feat['sleep_hours'] + eps)
    # Physical activity to counter screen time
    df_feat['physical_vs_screen'] = df_feat['physical_activity_hrs_per_week'] / (df_feat['screen_time_hrs_per_day'] + eps)
    
    # 4. Binning / Categorization (Useful for linear models like Logistic Regression)
    df_feat['is_sleep_deprived'] = (df_feat['sleep_hours'] < 6.0).astype(int)
    df_feat['is_highly_active'] = (df_feat['physical_activity_hrs_per_week'] >= 5.0).astype(int)
    df_feat['high_caffeine'] = (df_feat['caffeine_intake_mg_per_day'] > 250).astype(int)
    
    return df_feat
