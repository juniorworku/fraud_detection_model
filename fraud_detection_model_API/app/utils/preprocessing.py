import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Define the preprocessing pipeline (same as used during training)
numeric_features = ['purchase_value', 'age']
categorical_features = ['source', 'browser', 'sex', 'signup_hour', 'signup_day', 'purchase_hour', 'purchase_day']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

def preprocess_input(data):
    # Convert input data to DataFrame for preprocessing
    df = pd.DataFrame(data, index=[0])
    return preprocessor.transform(df)
