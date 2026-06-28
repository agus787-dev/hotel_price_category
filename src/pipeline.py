import pandas as pd
import numpy as np
import sys 

sys.path.append("..")

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from src.logger import Logger


class HotelPipeline(BaseEstimator):   

    def __init__(self, df: pd.DataFrame, target: str, log_file: str, model=None):
        self.df = df.copy()
        self.target = target
        self.model_algorithm = model
        self.model = None
        self.preprocessor = None
        self.log = Logger(log_file)
        self.log.info(f"RestaurantPipeline initialized | target: {target} | model: {type(model).__name__}")
        self.log.info(f"Dataset shape: {self.df.shape[0]} rows, {self.df.shape[1]} cols")

    def preprocessing(self):
        self.log.info("Preprocessing started")

        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        self.log.info(f"Target column: '{self.target}' | Features: {X.shape[1]} cols")

        num_col = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_col = X.select_dtypes(exclude=[np.number]).columns.tolist()
        self.log.info(f"Numeric columns ({len(num_col)}): {num_col}")
        self.log.info(f"Categorical columns ({len(cat_col)}): {cat_col}")

        # log missing values
        missing = X.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            self.log.warning(f"Missing values detected: {missing.to_dict()}")
        else:
            self.log.info("No missing values detected")

        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', MinMaxScaler())
        ])
        self.log.info("Numeric pipeline built: SimpleImputer(mean) → MinMaxScaler")

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        self.log.info("Categorical pipeline built: SimpleImputer(most_frequent) → OrdinalEncoder")

        self.preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, num_col),
            ('cat', categorical_pipeline, cat_col)
        ])

        self.log.info("Preprocessing finished")
        return X, y

    def fit(self):
        self.log.info("=" * 40)
        self.log.info("Training started")

        X, y = self.preprocessing()

        if self.model_algorithm is None:
            self.log.error("No model specified. Pass a scikit-learn estimator.")
            raise ValueError("No model specified. Pass a scikit-learn estimator.")

        self.log.info(f"Model algorithm: {type(self.model_algorithm).__name__}")

        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('estimator', self.model_algorithm)
        ])

        self.model.fit(X, y)

        self.log.info("Training finished")
        self.log.info(f"Train score: {self.model.score(X, y):.4f}")
        self.log.info("=" * 40)

        return self

    def predict(self, X=None):
        self.log.info("Predict started")

        if self.model is None:
            self.log.error("Model not fitted yet. Call fit() first.")
            raise ValueError("Model not fitted yet. Call fit() first.")

        if X is None:
            X = self.df.drop(columns=[self.target])
            self.log.info("No X provided — predicting on training data")
        else:
            self.log.info(f"Predicting on new data | shape: {X.shape}")

        predictions = self.model.predict(X)
        self.log.info(f"Predictions done | total: {len(predictions)}")
        return predictions

    def score(self, X=None, y=None):
        self.log.info("Score started")

        if self.model is None:
            self.log.error("Model not fitted yet. Call fit() first.")
            raise ValueError("Model not fitted yet. Call fit() first.")

        if X is None and y is None:
            X = self.df.drop(columns=[self.target])
            y = self.df[self.target]
            self.log.info("No X/y provided — scoring on training data")
        else:
            self.log.info(f"Scoring on provided data | shape: {X.shape}")

        result = self.model.score(X, y)
        self.log.info(f"Score result: {result:.4f}")
        return result