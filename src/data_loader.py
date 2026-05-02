import pandas as pd
import numpy as np
from src.config import DATA_PATH, TARGET_COL

class DataLoader:
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        
    def load_data(self):
        """Loads the raw dataset and maps target to binary."""
        df = pd.read_csv(self.data_path)
        
        # Map target variable: Y -> 1, N -> 0
        df[TARGET_COL] = df[TARGET_COL].map({'Y': 1, 'N': 0})
        
        return df

    def perform_basic_validation(self, df):
        """Returns basic statistics and info for EDA."""
        report = {
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
            "target_distribution": df[TARGET_COL].value_counts(normalize=True).to_dict()
        }
        return report
