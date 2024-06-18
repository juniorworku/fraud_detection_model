import pandas as pd
import os

# Load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Save data
def save_data(df, file_path):
    df.to_csv(file_path, index=False)

# Example function to process data
def process_data(df):
    # Add your data processing steps here
    return df

if __name__ == "__main__":
    raw_data_path = "data/raw/Fraud_Data.csv"
    processed_data_path = "data/processed/Fraud_Data_processed.csv"

    # Load raw data
    df = load_data(raw_data_path)
    
    # Process data
    df_processed = process_data(df)
    
    # Save processed data
    save_data(df_processed, processed_data_path)
