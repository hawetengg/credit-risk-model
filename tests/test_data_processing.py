# tests/test_data_processing.py
import sys
import os

print("\n--- Debugging sys.path (from test_data_processing.py) ---")
print("Current Working Directory:", os.getcwd())
print("PYTHONPATH environment variable:", os.environ.get('PYTHONPATH'))
print("sys.path contents:")
for i, p in enumerate(sys.path):
    print(f"  [{i}]: {p}")
print("--- End Debugging sys.path ---\n")

# The line that's failing, keep it here to see if the path is fixed before it runs:
from src.data_processing import load_data, extract_temporal_features

# ... rest of your test code

import pytest
import pandas as pd
from src.data_processing import load_data, extract_temporal_features

def test_load_data():
    """Test loading data from CSV."""
    df = load_data("data/raw/xente_data.csv")
    assert isinstance(df, pd.DataFrame), "Loaded data is not a DataFrame"
    assert not df.empty, "Loaded DataFrame is empty"

def test_extract_temporal_features():
    """Test extraction of temporal features."""
    sample_data = pd.DataFrame({
        'TransactionStartTime': ['2025-01-01 12:00:00', '2025-02-01 15:30:00']
    })
    result = extract_temporal_features(sample_data)
    assert 'TransactionHour' in result.columns, "TransactionHour not created"
    assert 'TransactionDay' in result.columns, "TransactionDay not created"
    assert result['TransactionHour'].iloc[0] == 12, "Incorrect TransactionHour"