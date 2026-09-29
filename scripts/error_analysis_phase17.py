import json
import numpy as np
import pandas as pd
import warnings
from scipy.stats import kruskal
warnings.filterwarnings('ignore')

def main():
    print("Loading data...")
    df = pd.read_csv('data/raw/train.csv')
    y_true = df['digital_academic_wellbeing_level']
    label_map = {'Low': 0, 'Medium': 1, 'High': 2}
    inv_label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    y_true_num = y_true.map(label_map).values
    
    print("Loading Phase 17 config and probabilities...")
    with open('experiments/phase17_ensemble_config.json', 'r') as f:
        config = json.load(f)
        
    weights = config['weights']
    mults = np.array(config['threshold_multipliers'])
    
    probas = np.load('experiments/phase17_oof_probas.npz')
    
    blended = np.zeros((len(df), 3))
    total_w = sum(weights.values())
    for name, w in weights.items():
        if w > 1e-4:
            blended += (w / total_w) * probas[name]
            
    adjusted = blended * mults
    y_pred_num = np.argmax(adjusted, axis=1)
    y_pred = [inv_label_map[p] for p in y_pred_num]
    
    df['pred'] = y_pred
    df['is_correct'] = (df['digital_academic_wellbeing_level'] == df['pred'])
    
    # Identify key misclassification paths
    df['error_type'] = df.apply(
        lambda row: f"True:{row['digital_academic_wellbeing_level']} -> Pred:{row['pred']}" if not row['is_correct'] else "Correct",
        axis=1
    )
    
    report_lines = []
    report_lines.append("# Phase 17 Ensemble Error Analysis\n")
    
    # 1. Confusion Summary
    report_lines.append("## Confusion Matrix Summary")
    cm_df = pd.crosstab(df['digital_academic_wellbeing_level'], df['pred'], rownames=['True'], colnames=['Pred'])
    report_lines.append(cm_df.to_markdown())
    report_lines.append("\n")
    
    # Most common errors
    errors = df[~df['is_correct']]
    error_counts = errors['error_type'].value_counts()
    report_lines.append("## Most Common Misclassifications")
    for err, count in error_counts.head(5).items():
        report_lines.append(f"- **{err}**: {count} cases ({count/len(df)*100:.1f}% of total data)")
    report_lines.append("\n")
    
    # 2. Feature Analysis of the Hardest Problem: True:Medium -> Pred:Low
    report_lines.append("## Feature Analysis: True:Medium -> Pred:Low vs True:Medium -> Pred:Medium")
    
    true_medium = df[df['digital_academic_wellbeing_level'] == 'Medium']
    med_correct = true_medium[true_medium['is_correct']]
    med_pred_low = true_medium[true_medium['pred'] == 'Low']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('is_correct', errors='ignore')
    
    diff_stats = []
    for col in numeric_cols:
        correct_vals = med_correct[col].dropna()
        error_vals = med_pred_low[col].dropna()
        if len(correct_vals) > 0 and len(error_vals) > 0:
            try:
                # Kruskal-Wallis H-test for statistical difference
                h_stat, p_val = kruskal(correct_vals, error_vals)
                if p_val < 0.05:
                    diff_stats.append({
                        'Feature': col,
                        'Correct Mean': correct_vals.mean(),
                        'Error (Pred Low) Mean': error_vals.mean(),
                        'p-value': p_val
                    })
            except:
                pass
                
    diff_df = pd.DataFrame(diff_stats)
    if not diff_df.empty:
        diff_df = diff_df.sort_values('p-value')
        report_lines.append("Features that are statistically different when the model mistakes a Medium for a Low:")
        report_lines.append(diff_df.head(10).to_markdown(index=False))
    else:
        report_lines.append("No statistically significant numeric feature differences found.")
    report_lines.append("\n")
    
    # 3. Feature Analysis: True:Medium -> Pred:High vs True:Medium -> Pred:Medium
    report_lines.append("## Feature Analysis: True:Medium -> Pred:High vs True:Medium -> Pred:Medium")
    med_pred_high = true_medium[true_medium['pred'] == 'High']
    
    diff_stats2 = []
    for col in numeric_cols:
        correct_vals = med_correct[col].dropna()
        error_vals = med_pred_high[col].dropna()
        if len(correct_vals) > 0 and len(error_vals) > 0:
            try:
                h_stat, p_val = kruskal(correct_vals, error_vals)
                if p_val < 0.05:
                    diff_stats2.append({
                        'Feature': col,
                        'Correct Mean': correct_vals.mean(),
                        'Error (Pred High) Mean': error_vals.mean(),
                        'p-value': p_val
                    })
            except:
                pass
                
    diff_df2 = pd.DataFrame(diff_stats2)
    if not diff_df2.empty:
        diff_df2 = diff_df2.sort_values('p-value')
        report_lines.append("Features that are statistically different when the model mistakes a Medium for a High:")
        report_lines.append(diff_df2.head(10).to_markdown(index=False))
    else:
        report_lines.append("No statistically significant numeric feature differences found.")
    
    with open('error_analysis_phase17.md', 'w') as f:
        f.write("\n".join(report_lines))
    print("Error analysis complete. Results saved to error_analysis_phase17.md")

if __name__ == '__main__':
    main()
