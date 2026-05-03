import pandas as pd
import numpy as np
from src.config import DATA_PATH, TARGET_COL

class DataLoader:
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        
    def load_data(self):
        """Loads the raw dataset and maps target to binary."""
        df = pd.read_csv(self.data_path)
        
        # Inject realistic but synthetic banking features since the dataset lacks them
        np.random.seed(42) # For reproducibility
        n = len(df)
        
        # Employment Stability (Years, right skewed)
        df['Employment_Stability'] = np.random.lognormal(mean=1.5, sigma=0.8, size=n).clip(0, 30).astype(int)
        
        # Existing Debt (correlated with ApplicantIncome)
        base_debt = df['ApplicantIncome'] * np.random.uniform(0.1, 2.5, n)
        df['Existing_Debt'] = np.where(base_debt < 1000, 0, base_debt).round(2)
        
        # Credit Utilization (0 to 100%, negatively correlated with Credit_History)
        # If Credit_History is 1, utilization tends to be lower (good). If 0, tends to be higher.
        utilization_good = np.random.normal(30, 15, n).clip(0, 100)
        utilization_bad = np.random.normal(80, 20, n).clip(0, 100)
        df['Credit_Utilization'] = np.where(df['Credit_History'] == 1.0, utilization_good, utilization_bad).round(1)
        
        # Drop Gender as requested by user
        if 'Gender' in df.columns:
            df.drop('Gender', axis=1, inplace=True)
        
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
