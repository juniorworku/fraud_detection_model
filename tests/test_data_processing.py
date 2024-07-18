# tests/test_data_processing.py

import pytest
from data_processing import DataProcessing

def test_load_data():
    processor = DataProcessing("path/to/test/dataset.csv")
    assert processor.data is not None

def test_preprocess_data():
    processor = DataProcessing("path/to/test/dataset.csv")
    preprocessed_data = processor.preprocess_data()
    assert preprocessed_data is not None
    assert 'categorical_feature_encoded' in preprocessed_data.columns

def test_scale_features():
    processor = DataProcessing("path/to/test/dataset.csv")
    preprocessed_data = processor.preprocess_data()
    scaled_data = processor.scale_features(['feature1', 'feature2'])
    assert scaled_data is not None
    assert scaled_data['feature1'].mean() == pytest.approx(0, abs=1e-2)
