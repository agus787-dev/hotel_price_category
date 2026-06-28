import pandas as pd
import sys
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.append("..")
from src.logger import Logger

class Preprocessor:
    def __init__(self, log_file: str):
        self.logger = Logger(log_file)
        self.scaler = StandardScaler()
        self.encoders = {}
        self.logger.info("Preprocessor initialized")
    
    def fillna_train(self, df):   
        self.logger.info("Starting fillna for training data")
        fill_values = {}
        df = df.copy()

        for col in df.columns:
            if df[col].dtype == 'object':
                fill_values[col] = df[col].mode()[0]
                df[col].fillna(fill_values[col], inplace=True)
            else:
                fill_values[col] = df[col].mean()
                df[col].fillna(fill_values[col], inplace=True)
            self.logger.debug(f"Filled NA for '{col}' with value: {fill_values[col]}")

        self.logger.info("Completed fillna for training data")
        return df, fill_values

    def fillna_test(self, df, fill_values):
        self.logger.info("Starting fillna for test data")
        df = df.copy()

        for col, value in fill_values.items():
            df[col].fillna(value, inplace=True)
            self.logger.debug(f"Filled NA for '{col}' with value: {value}")

        self.logger.info("Completed fillna for test data")
        return df
    
    def encode_train(self, df, threshold=2):
        self.logger.info(f"Starting encoding for training data with threshold={threshold}")
        df = df.copy()
        encoders = {}
        onehot_cols = []

        for col in df.columns:
            if df[col].dtype == 'object':
                if df[col].nunique() <= threshold:
                    onehot_cols.append(col)
                    dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                    df = pd.concat([df.drop(columns=col), dummies], axis=1)
                    self.logger.info(f"One-hot encoded '{col}'")
                else:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    encoders[col] = le
                    self.logger.info(f"Label encoded '{col}'")

        self.logger.info("Completed encoding for training data")
        return df, encoders, onehot_cols

    def encode_test(self, df, encoders, onehot_cols, train_columns):
        self.logger.info("Starting encoding for test data")
        df = df.copy()

        # Label encoding
        for col, le in encoders.items():
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            self.logger.debug(f"Label encoded test column '{col}'")

        # One-hot ONLY for known columns
        for col in onehot_cols:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            df = pd.concat([df.drop(columns=col), dummies], axis=1)
            self.logger.debug(f"One-hot encoded test column '{col}'")

        # Align columns
        df = df.reindex(columns=train_columns, fill_value=0)
        self.logger.info("Completed encoding for test data")

        return df
    
    def scale_train(self, df):
        self.logger.info("Starting scaling for training data")
        df = df.copy()
        scalers = {}

        for col in df.columns:
            if df[col].dtype != 'object':
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(df[[col]])
                scalers[col] = scaler
                self.logger.info(f"Scaled training column '{col}'")

        self.logger.info("Completed scaling for training data")
        return df, scalers

    def scale_test(self, df, scalers):
        self.logger.info("Starting scaling for test data")
        df = df.copy()

        for col in df.columns:
            if col in scalers:
                df[col] = scalers[col].transform(df[[col]])
                self.logger.info(f"Scaled test column '{col}'")

        self.logger.info("Completed scaling for test data")
        return df