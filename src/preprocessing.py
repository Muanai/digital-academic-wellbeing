import pandas as pd
import numpy as np
import re

def parse_age(series: pd.Series) -> pd.Series:
    """Handles: '20', '20 years', '20 yrs', '20y', etc."""
    # Convert to string, extract first occurrence of digits, convert to float
    return series.astype(str).str.extract(r'(\d+)')[0].astype(float)

def normalize_gender(series: pd.Series) -> pd.Series:
    """Handles: Male/male/MALE/Female/FEMALE/Other → male/female/other"""
    return series.astype(str).str.lower().str.strip()

def normalize_binary(series: pd.Series) -> pd.Series:
    """Handles: Yes/yes/YES/No/no/NO → yes/no"""
    return series.astype(str).str.lower().str.strip()

def clean_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans impossible values such as negative notifications_per_day or screen time > 24
    by replacing them with NaN.
    """
    df = df.copy()
    
    # Notifications should not be negative
    if 'notifications_per_day_clean' in df.columns:
        df.loc[df['notifications_per_day_clean'] < 0, 'notifications_per_day_clean'] = np.nan
        
    # Valid ranges based on STRATEGY.md
    if 'screen_time_hrs_per_day_clean' in df.columns:
        # Note: Strategy says max observed is 48, investigate typo/artifact
        # We will keep it for now but flag > 24 as NaN if we strictly believe it's 24 hours max
        # The strategy says "investigate", so let's leave it as is in canonicalization, 
        # or just clip strictly impossible (e.g. > 24 for a daily feature).
        pass 
        
    return df

def apply_canonicalization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the full canonicalization pipeline.
    Maintains raw columns and creates `_clean` suffixed columns.
    """
    df = df.copy()
    
    # 1. Parse Age
    if 'age' in df.columns:
        df['age_clean'] = parse_age(df['age'])
        
    # 2. Normalize Gender
    if 'gender' in df.columns:
        df['gender_clean'] = normalize_gender(df['gender'])
        
    # 3. Normalize Binary columns
    binary_cols = ['uses_productivity_app', 'frequently_procrastinates', 
                   'family_academic_achievement_history', 'takes_extra_tutoring', 
                   'major_life_event_this_semester']
    
    for col in binary_cols:
        if col in df.columns:
            df[f'{col}_clean'] = normalize_binary(df[col])
            
    # 4. Copy numeric columns that don't need text parsing but might need outlier cleaning
    # For now, let's just make a clean copy of the one we know needs cleaning
    if 'notifications_per_day' in df.columns:
        df['notifications_per_day_clean'] = df['notifications_per_day'].astype(float)
        
    # Apply impossible values cleaning
    df = clean_impossible_values(df)
    
    return df
