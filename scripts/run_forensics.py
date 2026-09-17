import pandas as pd
import numpy as np
import os
import sys
from scipy.stats import ks_2samp, chi2_contingency
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import apply_canonicalization

def main():
    print("Starting Forensics...")
    data_dir = Path("data/raw")
    
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    
    if not train_path.exists():
        print("train.csv not found!")
        return
        
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path) if test_path.exists() else None
    
    # 1. Apply Canonicalization
    df_train_clean = apply_canonicalization(df_train)
    df_test_clean = apply_canonicalization(df_test) if df_test is not None else None
    
    report_lines = []
    report_lines.append("# Phase 1 Forensics Report")
    report_lines.append(f"**Train rows**: {len(df_train)} | **Test rows**: {len(df_test) if df_test is not None else 'N/A'}")
    
    # 1.1 ID Investigation
    report_lines.append("\n## 1.1 ID Investigation")
    # Extract numeric part of ID
    df_train_clean['id_num'] = df_train_clean['respondent_id'].str.extract(r'(\d+)').astype(float)
    if df_test_clean is not None:
        df_test_clean['id_num'] = df_test_clean['respondent_id'].str.extract(r'(\d+)').astype(float)
        
        train_id_min, train_id_max = df_train_clean['id_num'].min(), df_train_clean['id_num'].max()
        test_id_min, test_id_max = df_test_clean['id_num'].min(), df_test_clean['id_num'].max()
        
        report_lines.append(f"- **Train ID Range**: {train_id_min} to {train_id_max}")
        report_lines.append(f"- **Test ID Range**: {test_id_min} to {test_id_max}")
        if train_id_max < test_id_min or test_id_max < train_id_min:
            report_lines.append("-> **Conclusion**: IDs do NOT overlap. They are split by time or purely sequential.")
        else:
            report_lines.append("-> **Conclusion**: IDs OVERLAP. Be careful of leakage if ID is used as feature.")
            
    # Target distribution across ID bins in train
    if 'digital_academic_wellbeing_level' in df_train_clean.columns:
        df_train_clean['id_bin'] = pd.qcut(df_train_clean['id_num'], q=5, duplicates='drop')
        target_by_bin = df_train_clean.groupby('id_bin')['digital_academic_wellbeing_level'].value_counts(normalize=True).unstack().fillna(0)
        report_lines.append("\n**Target Distribution by ID Quintiles:**\n")
        # Custom format to avoid tabulate dependency
        target_by_bin_str = "| id_bin | " + " | ".join(target_by_bin.columns) + " |\n"
        target_by_bin_str += "|---|" + "|".join(["---" for _ in target_by_bin.columns]) + "|\n"
        for idx, row in target_by_bin.iterrows():
            target_by_bin_str += f"| {idx} | " + " | ".join([f"{v:.3f}" for v in row.values]) + " |\n"
        report_lines.append(target_by_bin_str)
        
    # 1.2 Canonicalization Audit
    report_lines.append("\n## 1.2 Canonicalization Audit")
    report_lines.append("\n**Age noise example (Raw vs Clean):**")
    age_compare = df_train_clean[['age', 'age_clean']].drop_duplicates().head(10)
    age_str = "| age | age_clean |\n|---|---|\n"
    for _, row in age_compare.iterrows():
        age_str += f"| {row['age']} | {row['age_clean']} |\n"
    report_lines.append("\n" + age_str)
    
    report_lines.append("\n**Gender noise example (Raw vs Clean):**")
    gender_compare = df_train_clean[['gender', 'gender_clean']].drop_duplicates().head(10)
    gender_str = "| gender | gender_clean |\n|---|---|\n"
    for _, row in gender_compare.iterrows():
        gender_str += f"| {row['gender']} | {row['gender_clean']} |\n"
    report_lines.append("\n" + gender_str)

    # 1.3 Outlier Audit
    report_lines.append("\n## 1.3 Outlier Audit (Train)")
    report_lines.append("\n| Feature | Min | Max | Mean | 99th Pct |")
    report_lines.append("|---------|-----|-----|------|----------|")
    numeric_cols = df_train_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ['id_num']: continue
        v_min = df_train_clean[col].min()
        v_max = df_train_clean[col].max()
        v_mean = df_train_clean[col].mean()
        v_99 = df_train_clean[col].quantile(0.99)
        report_lines.append(f"| `{col}` | {v_min:.2f} | {v_max:.2f} | {v_mean:.2f} | {v_99:.2f} |")
        
    # 1.4 Missingness Audit
    report_lines.append("\n## 1.4 Missingness Audit (Train)")
    missing_counts = df_train_clean.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0].index.tolist()
    
    if not missing_cols:
        report_lines.append("\nNo missing values found in train set!")
    else:
        report_lines.append("\n| Feature | Missing Count | Missing % | Target Correlation (Chi2 p-value) |")
        report_lines.append("|---------|---------------|-----------|-----------------------------------|")
        
        target = df_train_clean['digital_academic_wellbeing_level']
        for col in missing_cols:
            m_count = missing_counts[col]
            m_pct = (m_count / len(df_train_clean)) * 100
            
            # Chi-square test between missingness indicator and target
            is_missing = df_train_clean[col].isnull().astype(int)
            contingency = pd.crosstab(is_missing, target)
            chi2, p, dof, ex = chi2_contingency(contingency)
            
            # If p < 0.05, missingness is informative!
            sig = " (Significant!)" if p < 0.05 else ""
            report_lines.append(f"| `{col}` | {m_count} | {m_pct:.2f}% | {p:.4e}{sig} |")

    # 1.5 Distribution Drift
    report_lines.append("\n## 1.5 Train/Test Distribution Drift (KS Test)")
    if df_test_clean is None:
        report_lines.append("No test data found to compare drift.")
    else:
        report_lines.append("\n| Feature | KS Statistic | P-Value | Risk |")
        report_lines.append("|---------|--------------|---------|------|")
        for col in numeric_cols:
            if col not in df_test_clean.columns or col == 'id_num': continue
            
            train_vals = df_train_clean[col].dropna()
            test_vals = df_test_clean[col].dropna()
            
            if len(train_vals) == 0 or len(test_vals) == 0:
                continue
                
            ks_stat, p_val = ks_2samp(train_vals, test_vals)
            risk = "High Drift Risk" if p_val < 0.05 else "Safe"
            report_lines.append(f"| `{col}` | {ks_stat:.4f} | {p_val:.4e} | {risk} |")

    # 1.6 Duplicate Hunt
    report_lines.append("\n## 1.6 Duplicate Hunt")
    
    # Drop ID for duplicate check
    dup_check_train = df_train.drop(columns=['respondent_id'])
    n_dupes_train = dup_check_train.duplicated().sum()
    report_lines.append(f"- **Exact Duplicate rows in Train (excluding ID)**: {n_dupes_train}")
    
    if df_test is not None:
        dup_check_test = df_test.drop(columns=['respondent_id'])
        n_dupes_test = dup_check_test.duplicated().sum()
        report_lines.append(f"- **Exact Duplicate rows in Test (excluding ID)**: {n_dupes_test}")
        
        # Cross-duplicates
        train_hashes = pd.util.hash_pandas_object(dup_check_train, index=False)
        test_hashes = pd.util.hash_pandas_object(dup_check_test, index=False)
        cross_dupes = test_hashes.isin(train_hashes).sum()
        report_lines.append(f"- **Test rows that exactly match a Train row**: {cross_dupes}")

    # Write report
    report_path = r"C:\Users\user\.gemini\antigravity-ide\brain\61d1d5ec-f855-42cb-8916-4ccc318a689b\phase1_forensics_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
