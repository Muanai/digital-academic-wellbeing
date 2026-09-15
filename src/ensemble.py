import numpy as np

def simple_average(predictions_list: list) -> np.ndarray:
    """
    Simple mean ensembling of multiple predictions.
    """
    return np.mean(predictions_list, axis=0)

def weighted_average(predictions_list: list, weights: list) -> np.ndarray:
    """
    Weighted mean ensembling.
    """
    return np.average(predictions_list, axis=0, weights=weights)
