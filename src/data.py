import pandas as pd
import polars as pl
from pathlib import Path

def load_data(data_path: str, use_polars: bool = False):
    """
    Load raw data from csv or parquet.
    """
    path = Path(data_path)
    if use_polars:
        train = pl.read_csv(path / "train.csv") if (path / "train.csv").exists() else None
        test = pl.read_csv(path / "test.csv") if (path / "test.csv").exists() else None
    else:
        train = pd.read_csv(path / "train.csv") if (path / "train.csv").exists() else None
        test = pd.read_csv(path / "test.csv") if (path / "test.csv").exists() else None
        
    return train, test
