import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy.optimize import minimize

def get_cv_strategy(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    """
    Returns the strict CV strategy for the competition (Phase 2 constraint).
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def evaluate_model(model, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold,
                   sample_weight: np.ndarray = None):
    """
    Evaluates a model using the provided CV strategy.
    Returns:
        overall_score: OOF Macro-F1 (float)
        oof_preds:     OOF hard predictions (pd.Series, same index as y)
        oof_proba:     OOF probabilities    (np.ndarray, shape [n_samples, n_classes])
                       None if model has no predict_proba.
    
    Phase 3.5 update:
    - OOF probabilities returned for ensemble / threshold tuning / stacking.
    - sample_weight support: pass pre-computed weights for XGBoost balancing.
      Weights are sliced per fold and passed as classifier__sample_weight.
    """
    # Create empty Series for OOF hard predictions with same index as y
    oof_preds = pd.Series(index=y.index, dtype=y.dtype)
    
    n_classes = y.nunique()
    oof_proba = np.full((len(y), n_classes), np.nan)
    
    fold_scores = []
    
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series) else y
    
    has_proba = hasattr(model, 'predict_proba')
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr)):
        X_train = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X_arr[train_idx]
        y_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y_arr[train_idx]
        
        X_val = X.iloc[val_idx] if isinstance(X, pd.DataFrame) else X_arr[val_idx]
        y_val = y.iloc[val_idx] if isinstance(y, pd.Series) else y_arr[val_idx]
        
        # Pass sample_weight if provided (XGBoost class balancing workaround)
        if sample_weight is not None:
            sw_train = sample_weight[train_idx]
            model.fit(X_train, y_train, classifier__sample_weight=sw_train)
        else:
            model.fit(X_train, y_train)
        
        preds = model.predict(X_val)
        
        # Flatten predictions if model (like CatBoost) returns 2D array
        if preds.ndim > 1:
            preds = preds.ravel()
            
        # Save OOF hard predictions
        oof_preds.iloc[val_idx] = preds
        
        # Save OOF probabilities if available
        if has_proba:
            proba = model.predict_proba(X_val)
            oof_proba[val_idx] = proba
        
        score = f1_score(y_val, preds, average='macro')
        fold_scores.append(score)
        print(f"  Fold {fold+1} Macro-F1: {score:.4f}")
        
    overall_score = f1_score(y, oof_preds, average='macro')
    print(f"  Overall OOF Macro-F1: {overall_score:.4f}")
    
    return overall_score, oof_preds, (oof_proba if has_proba else None)


def optimize_thresholds(y_true, oof_proba: np.ndarray) -> tuple:
    """
    Optimizes class probability multipliers to maximize Macro-F1 score.
    Returns:
        best_f1: Float (optimized Macro-F1 score)
        best_weights: Array of multipliers
        best_preds: pd.Series of the optimized hard predictions
    """
    if oof_proba is None:
        return None, None, None
        
    y_true_arr = y_true.values if isinstance(y_true, pd.Series) else y_true
    
    def loss(weights):
        w_proba = oof_proba * weights
        preds = np.argmax(w_proba, axis=1)
        return -f1_score(y_true_arr, preds, average='macro')
        
    res = minimize(loss, [1.0, 1.0, 1.0], method='Nelder-Mead')
    best_weights = res.x
    
    w_proba = oof_proba * best_weights
    best_preds = np.argmax(w_proba, axis=1)
    best_f1 = -res.fun
    
    idx = y_true.index if hasattr(y_true, 'index') else None
    return best_f1, best_weights, pd.Series(best_preds, index=idx)

def optimize_regression_thresholds(y_true, y_pred_cont: np.ndarray) -> tuple:
    """
    Optimizes two thresholds [t1, t2] to convert continuous regression predictions
    into discrete class labels [0, 1, 2] to maximize Macro-F1 score.
    Returns:
        best_f1: Float (optimized Macro-F1 score)
        best_thresholds: Array of [t1, t2]
        best_preds: pd.Series of the optimized hard predictions
    """
    if y_pred_cont is None:
        return None, None, None
        
    y_true_arr = y_true.values if isinstance(y_true, pd.Series) else y_true
    
    def loss(thresholds):
        t1, t2 = thresholds
        if t1 >= t2:
            return 0.0
            
        preds = np.zeros_like(y_pred_cont)
        preds[y_pred_cont >= t1] = 1
        preds[y_pred_cont >= t2] = 2
        return -f1_score(y_true_arr, preds, average='macro')
        
    res = minimize(loss, [0.5, 1.5], method='Nelder-Mead')
    best_thresholds = res.x
    
    t1, t2 = best_thresholds
    best_preds = np.zeros_like(y_pred_cont)
    best_preds[y_pred_cont >= t1] = 1
    best_preds[y_pred_cont >= t2] = 2
    
    best_f1 = -res.fun
    
    idx = y_true.index if hasattr(y_true, 'index') else None
    return best_f1, best_thresholds, pd.Series(best_preds, index=idx)


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
