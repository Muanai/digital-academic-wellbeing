import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def get_cv_strategy(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    """
    Returns the strict CV strategy for the competition (Phase 2 constraint).
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def evaluate_model(model, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold):
    """
    Evaluates a model using the provided CV strategy.
    Returns the OOF (Out-Of-Fold) predictions and the overall Macro-F1 score.
    """
    # Create empty Series for OOF predictions with same index as y
    oof_preds = pd.Series(index=y.index, dtype=y.dtype)
    
    fold_scores = []
    
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series) else y
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr)):
        X_train = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X_arr[train_idx]
        y_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y_arr[train_idx]
        
        X_val = X.iloc[val_idx] if isinstance(X, pd.DataFrame) else X_arr[val_idx]
        y_val = y.iloc[val_idx] if isinstance(y, pd.Series) else y_arr[val_idx]
        
        model.fit(X_train, y_train)
        
        preds = model.predict(X_val)
        
        # Flatten predictions if model (like CatBoost) returns 2D array
        if preds.ndim > 1:
            preds = preds.ravel()
            
        # Save OOF predictions
        oof_preds.iloc[val_idx] = preds
        
        score = f1_score(y_val, preds, average='macro')
        fold_scores.append(score)
        print(f"  Fold {fold+1} Macro-F1: {score:.4f}")
        
    overall_score = f1_score(y, oof_preds, average='macro')
    print(f"  Overall OOF Macro-F1: {overall_score:.4f}")
    
    return overall_score, oof_preds

def generate_error_analysis(y_true, y_pred, model_name: str, output_dir: str):
    """
    Generates a confusion matrix plot and returns indices of misclassified samples.
    """
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    # Target map: {'Low': 0, 'Medium': 1, 'High': 2}
    labels = ['Low (0)', 'Medium (1)', 'High (2)']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    
    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    
    # 2. Find misclassified indices
    # We return a boolean mask or indices where true != pred
    misclassified_mask = y_true != y_pred
    
    return cm_path, misclassified_mask
