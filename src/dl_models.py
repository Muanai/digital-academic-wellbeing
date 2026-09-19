import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
import warnings
warnings.filterwarnings("ignore")

class TabularResNetWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, numeric_features, categorical_features, target_col='target', random_state=42):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_col = target_col
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        df = X.copy()
        df[self.target_col] = np.array(y)
        
        # Simple imputation for deep learning
        for col in self.categorical_features:
            df[col] = df[col].fillna("Missing").astype(str)
        for col in self.numeric_features:
            df[col] = df[col].fillna(0).astype('float32')
            
        data_config = DataConfig(
            target=[self.target_col],
            continuous_cols=self.numeric_features,
            categorical_cols=self.categorical_features,
        )
        trainer_config = TrainerConfig(
            auto_lr_find=False,
            batch_size=64,
            max_epochs=15,
            seed=self.random_state,
            accelerator="cpu",
            progress_bar="none",
            checkpoints=None
        )
        optimizer_config = OptimizerConfig()
        
        model_config = CategoryEmbeddingModelConfig(
            task="classification",
            learning_rate=1e-3,
            layers="128-64",
            activation="ReLU",
            seed=self.random_state
        )
        
        self.model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            verbose=0
        )
        
        self.model.fit(train=df)
        return self

    def predict_proba(self, X):
        df = X.copy()
        for col in self.categorical_features:
            df[col] = df[col].fillna("Missing").astype(str)
        for col in self.numeric_features:
            df[col] = df[col].fillna(0).astype('float32')
            
        preds = self.model.predict(df)
        
        # Extract probabilities from the returned dataframe
        cols = [f"{self.target_col}_probability_{i}" for i in range(3)]
        if all(c in preds.columns for c in cols):
            return preds[cols].values
        else:
            prob_cols = [c for c in preds.columns if "probability" in c.lower()]
            return preds[prob_cols].values

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
