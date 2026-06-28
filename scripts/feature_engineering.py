
import pandas as pd
import numpy as np
import sys
import os

sys.path.append("..")
from src.preprocess import Preprocessor
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer


loader=DataLoader("../data/merged/hotels.csv", "feature_engineering")
prep = Preprocessor("feature_engineering")
fe = FeatureEngineer()

# Loading data
df = loader.load()

# Splitting
x_train, x_test, y_train, y_test = loader.split(target_column="price_category")


# Handling missing values
x_train_filled, fill_values = prep.fillna_train(df=x_train)
x_test_filled = prep.fillna_test(df=x_test, fill_values=fill_values)

# Creaing new features
x_train_fe=fe.add_features(x_train_filled)
x_test_fe=fe.add_features(x_test_filled)

# Encoding
x_train_encoded, encoders, one_hot_cols = prep.encode_train(df=x_train_filled)
x_test_encoded = prep.encode_test(
  df=x_test_filled, 
  encoders=encoders, 
  onehot_cols=one_hot_cols, 
  train_columns=x_train_encoded.columns
)

# Skewness
fe.check_skewness(x_train_encoded)
skewed_features = fe.fix_skewness(x_train_encoded)

X_train_transformed = x_train_encoded.copy()
X_test_transformed = x_test_encoded.copy()

for col in skewed_features:
    if (X_train_transformed[col] >= 0).all():
        
        X_train_transformed[col] = np.log1p(X_train_transformed[col])
        X_test_transformed[col] = np.log1p(X_test_transformed[col])


# Scaling
x_train_scaled, scalers = prep.scale_train(df=X_train_transformed)
x_test_scaled = prep.scale_test(df=X_test_transformed, scalers=scalers)

# Training
fe.ovo_smote(x_train_scaled, y_train, x_test_scaled, y_test)







