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
    def add_geolocation_features(self, ip_to_country_df):
        self.data['ip_int'] = self.data['ip_address'].apply(self.ip_to_int)
        self.data = pd.merge(self.data, ip_to_country_df, how='left', left_on='ip_int', right_on='lower_bound_ip_address')
        return self.data

    def ip_to_int(self, ip):
        o = list(map(int, ip.split('.')))
        return (16777216 * o[0]) + (65536 * o[1]) + (256 * o[2]) + o[3]
