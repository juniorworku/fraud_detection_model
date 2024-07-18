# Refactored data_processing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataProcessing:
    def __init__(self, file_path):
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        return pd.read_csv(file_path)

    def preprocess_data(self):
        # Handle missing values
        self.data = self.data.dropna()
        # Encode categorical features
        self.data = pd.get_dummies(self.data)
        return self.data

    def scale_features(self, columns):
        scaler = StandardScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])
        return self.data
