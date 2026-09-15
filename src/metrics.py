import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

def compute_metrics(y_true, y_pred, task_type='regression'):
    """
    Compute validation metrics.
    """
    if task_type == 'regression':
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return {"rmse": rmse}
    else:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        return {"accuracy": acc, "f1_macro": f1}
