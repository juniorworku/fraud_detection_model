import unittest
import pandas as pd
import numpy as np
from data_processing import (
    handle_missing_values, clean_data, convert_ip_to_int,
    merge_datasets, feature_engineering, normalize_and_scale, 
    encode_categorical_features
)

class TestDataProcessing(unittest.TestCase):
    
    def setUp(self):
        self.fraud_data = pd.DataFrame({
            'user_id': [1, 2, 2, 3, 4, 5],
            'purchase_time': ['2021-07-16 12:01:00', '2021-07-16 12:05:00', 
                              '2021-07-16 12:05:00', '2021-07-16 12:10:00', 
                              '2021-07-16 12:20:00', '2021-07-16 12:30:00'],
            'ip_address': ['192.168.1.1', '192.168.1.2', '192.168.1.2', 
                           '192.168.1.3', '192.168.1.4', '192.168.1.5'],
            'purchase_value': [100, 200, 200, 150, 100, 250],
            'class': [0, 1, 1, 0, 0, 1],
            'source': ['web', 'web', 'mobile', 'web', 'mobile', 'web'],
            'browser': ['chrome', 'safari', 'chrome', 'firefox', 'chrome', 'safari'],
            'sex': ['M', 'F', 'M', 'M', 'F', 'F'],
            'country': ['US', 'US', 'CA', 'US', 'CA', 'US']
        })

        self.ip_country = pd.DataFrame({
            'lower_bound_ip_address': ['192.168.1.0', '192.168.1.4'],
            'upper_bound_ip_address': ['192.168.1.3', '192.168.1.7'],
            'country': ['US', 'CA']
        })

        self.credit_card = pd.DataFrame({
            'Time': [1, 2, 3, 4, 5],
            'Amount': [100, 200, 150, 300, 250],
            'Class': [0, 1, 0, 1, 0]
        })
    
    def test_handle_missing_values(self):
        df = self.fraud_data.copy()
        df.loc[0, 'purchase_value'] = np.nan
        df = handle_missing_values(df)
        self.assertFalse(df.isnull().values.any())
    
    def test_clean_data(self):
        df = self.fraud_data.copy()
        df = clean_data(df)
        self.assertEqual(len(df), len(df.drop_duplicates()))
    
    def test_convert_ip_to_int(self):
        df = self.fraud_data.copy()
        df = convert_ip_to_int(df, ['ip_address'])
        self.assertTrue(pd.api.types.is_integer_dtype(df['ip_address']))
    
    def test_merge_datasets(self):
        fraud_data = convert_ip_to_int(self.fraud_data.copy(), ['ip_address'])
        ip_country = convert_ip_to_int(self.ip_country.copy(), ['lower_bound_ip_address', 'upper_bound_ip_address'])
        merged_data = merge_datasets(fraud_data, ip_country)
        self.assertIn('country', merged_data.columns)
    
    def test_feature_engineering(self):
        df = feature_engineering(self.fraud_data.copy())
        self.assertIn('transaction_frequency', df.columns)
        self.assertIn('hour_of_day', df.columns)
        self.assertIn('day_of_week', df.columns)
    
    def test_normalize_and_scale(self):
        fraud_data, credit_card = normalize_and_scale(self.fraud_data.copy(), self.credit_card.copy())
        self.assertTrue(np.all(fraud_data['purchase_value'] <= 1) and np.all(fraud_data['purchase_value'] >= -1))
        self.assertTrue(np.all(credit_card['Amount'] <= 1) and np.all(credit_card['Amount'] >= -1))
    
    def test_encode_categorical_features(self):
        df = encode_categorical_features(self.fraud_data.copy())
        self.assertNotIn('source', df.columns)
        self.assertNotIn('browser', df.columns)
        self.assertNotIn('sex', df.columns)
        self.assertNotIn('country', df.columns)
        self.assertTrue(df.shape[1] > self.fraud_data.shape[1])

if __name__ == '__main__':
    unittest.main()
