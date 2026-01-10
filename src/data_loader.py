import pandas as pd

from src.config import DATA_PATH

def load_data(path):
    """Load the customer churn dataset from the specified path."""
    df = pd.read_csv(path)
    print(f"Data loaded successfully from {path}.")
    return df