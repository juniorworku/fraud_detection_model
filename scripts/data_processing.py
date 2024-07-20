import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import ipaddress

def load_datasets():
    fraud_data = pd.read_csv('Fraud_Data.csv')
    ip_country = pd.read_csv('IpAddress_to_Country.csv')
    credit_card = pd.read_csv('creditcard.csv')
    return fraud_data, ip_country, credit_card

def handle_missing_values(df):
    return df.dropna()

def clean_data(df):
    df.drop_duplicates(inplace=True)
    return df

def convert_ip_to_int(df, columns):
    for col in columns:
        df[col] = df[col].apply(lambda x: int(ipaddress.ip_address(x)))
    return df

def merge_datasets(fraud_data, ip_country):
    return pd.merge(fraud_data, ip_country, how='left', left_on='ip_address', right_on='lower_bound_ip_address')

def feature_engineering(fraud_data):
    fraud_data['transaction_frequency'] = fraud_data.groupby('user_id')['purchase_time'].transform('count')
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
    fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
    fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
    return fraud_data

def normalize_and_scale(fraud_data, credit_card):
    scaler = StandardScaler()
    fraud_data[['purchase_value', 'transaction_frequency']] = scaler.fit_transform(fraud_data[['purchase_value', 'transaction_frequency']])
    credit_card['Amount'] = scaler.fit_transform(credit_card[['Amount']])
    return fraud_data, credit_card

def encode_categorical_features(fraud_data):
    encoder = OneHotEncoder()
    categorical_features = ['source', 'browser', 'sex', 'country']
    encoded_features = encoder.fit_transform(fraud_data[categorical_features])
    fraud_data = fraud_data.drop(columns=categorical_features)
    fraud_data = pd.concat([fraud_data, pd.DataFrame(encoded_features.toarray())], axis=1)
    return fraud_data

def main():
    fraud_data, ip_country, credit_card = load_datasets()
    fraud_data = handle_missing_values(fraud_data)
    credit_card = handle_missing_values(credit_card)
    fraud_data = clean_data(fraud_data)
    credit_card = clean_data(credit_card)
    ip_country = convert_ip_to_int(ip_country, ['lower_bound_ip_address', 'upper_bound_ip_address'])
    fraud_data = convert_ip_to_int(fraud_data, ['ip_address'])
    fraud_data = merge_datasets(fraud_data, ip_country)
    fraud_data = feature_engineering(fraud_data)
    fraud_data, credit_card = normalize_and_scale(fraud_data, credit_card)
    fraud_data = encode_categorical_features(fraud_data)
    fraud_data.to_csv('processed_fraud_data.csv', index=False)
    credit_card.to_csv('processed_credit_card.csv', index=False)

if __name__ == "__main__":
    main()
