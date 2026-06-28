import pandas as pd
import os
import sys

sys.path.append("..")

from sklearn.model_selection import train_test_split
from src.logger import Logger


class DataLoader:
    def __init__(self, path: str, log_file: str):
      self.path = path
      self.log_file = log_file
      self.df   = None
      self.log = Logger(log_file)
        
        

    def load(self):
        self.df = pd.read_csv(self.path)
        # self.df = self.df.dropna(axis=0)  # Drop rows where all values are NA
        # self.df = klib.clean_column_names(self.df)
        self.log.info(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} cols")
        return self.df

    def split(self, target_column, test_size=0.2, to_drop=[]):
        X = self.df.drop(columns=[target_column] + to_drop)
        y = self.df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        self.log.info(f"Train and test split => Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test