from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

def get_cv_strategy(strategy: str, n_splits: int, random_state: int = 42):
    """
    Get Cross-Validation strategy.
    """
    if strategy == 'kfold':
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif strategy == 'stratified':
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif strategy == 'group':
        return GroupKFold(n_splits=n_splits)
    else:
        raise ValueError(f"CV strategy {strategy} not supported.")
