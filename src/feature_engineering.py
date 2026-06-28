
import pandas as pd
import sys
import joblib

from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score


sys.path.append("..")
from src.logger import Logger

ROOT_DIR = Path(__file__).resolve().parent.parent



class FeatureEngineer:
    def __init__(self):
        self.skewed_cols  = []
        self.logger = Logger("feature_engineering")
        self.logger.info("FeatureEngineer initialized")

        self.output_dir_path = ROOT_DIR / "models" / "engineered"
        self.output_dir_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Output directory ready: {self.output_dir_path}")

    def add_features(self, df):
        self.logger.info("Starting feature engineering")
        df = df.copy()
        self.logger.info(f"Input dataframe shape: {df.shape}")

        # Distance category
        def distance_category(distance):
            if distance < 1:
                return "Very Close"
            elif distance < 3:
                return "Close"
            elif distance < 7:
                return "Medium"
            else:
                return "Far"

        df["distance_category"] = df["distance_to_center"].apply(distance_category)

        # City popularity
        city_counts = df["city"].value_counts()

        df["city_popularity"] = df["city"].map(city_counts)

        # Adults category
        def adults_category(n):
            if n == 1:
                return "Solo"
            elif n == 2:
                return "Couple"
            elif n <= 4:
                return "Family"
            else:
                return "Group"

        df["adults_category"] = df["adults"].apply(adults_category)



        added_features = [
            "distance_category",
            "city_popularity",
            "adults_category"
        ]
        self.logger.info(f"Added engineered features: {added_features}")
        self.logger.info(f"Output dataframe shape: {df.shape}")
        self.logger.info("Completed feature engineering")
        return df

    def fix_skewness(self, df, threshold=0.4):
        self.logger.info(f"Checking skewness with threshold {threshold}")
        df = df.copy()
        numeric_df = df.select_dtypes(include="number")
        self.logger.info(f"Numeric feature count for skewness check: {numeric_df.shape[1]}")
        skewness = numeric_df.skew()
        skewed_features = skewness[(skewness >= threshold)].sort_values(ascending=False)
        self.logger.info(f"Found {len(skewed_features)} skewed numeric features above threshold: {skewed_features.index.tolist()}")
        return skewed_features.index.tolist()

    def check_skewness(self, df):
        self.logger.info("Checking skewness for numeric features")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        self.logger.info(f"Numeric columns found for skewness check: {num_cols}")
        skew_df = df[num_cols].skew().abs().sort_values(ascending=False)
        self.logger.info(f"Skewness summary (top 5):\n{skew_df.head(5).to_string()}")
        return skew_df
    
    def ovo_smote(self, x_train, y_train, x_test, y_test):
        self.logger.info("=" * 50)
        self.logger.info("Starting One-vs-One + SMOTE training")

        # SMOTE
        self.logger.info("Applying SMOTE oversampling")
        smote = SMOTE(random_state=42, k_neighbors=1)
        X_smote, y_smote = smote.fit_resample(x_train, y_train)

        self.logger.info(
            f"SMOTE completed | Original: {x_train.shape[0]} samples | "
            f"Resampled: {X_smote.shape[0]} samples"
        )

        # Model
        self.logger.info("Training One-vs-One Logistic Regression model")
        ovo_model_smote = OneVsOneClassifier(
            LogisticRegression(max_iter=1000, random_state=42)
        )

        ovo_model_smote.fit(X_smote, y_smote)

        # Prediction
        self.logger.info("Making predictions")
        y_pred = ovo_model_smote.predict(x_test)

        # Accuracy
        acc_score = accuracy_score(y_test, y_pred)
        self.logger.info(f"Accuracy: {acc_score:.4f}")

        # Save model
        output_file = self.output_dir_path / "LogisticRegression.joblib"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(ovo_model_smote, output_file)

        self.logger.info(f"Model saved: {output_file}")
        self.logger.info("=" * 50)

        return acc_score



    