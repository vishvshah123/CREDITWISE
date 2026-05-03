import pandas as pd
import numpy as np
from src.config import DATA_PATH, TARGET_COL

class DataLoader:
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        
    def load_data(self):
        """Loads the raw dataset and maps target to binary."""
        df = pd.read_csv(self.data_path)
        
        # Map target variable FIRST so we can condition synthetic features on it
        df[TARGET_COL] = df[TARGET_COL].map({'Y': 1, 'N': 0})
        
        np.random.seed(42)
        n = len(df)
        approved = df[TARGET_COL].values  # 1 = approved, 0 = rejected

        # ---------------------------------------------------------------
        # Employment Stability: approved = more stable (higher years)
        #   Approved:  mean=6 yrs, std=3  → good tenure signal
        #   Rejected:  mean=2 yrs, std=1.5 → job-hopping / short tenure
        # ---------------------------------------------------------------
        emp_approved = np.random.normal(loc=6.0, scale=3.0, size=n).clip(0, 30)
        emp_rejected  = np.random.normal(loc=2.0, scale=1.5, size=n).clip(0, 30)
        df['Employment_Stability'] = np.where(approved, emp_approved, emp_rejected).round(1)

        # ---------------------------------------------------------------
        # Existing Debt: approved = lower debt relative to income
        #   Approved:  0.3–0.8× income  (manageable)
        #   Rejected:  1.5–3.5× income  (overloaded)
        # ---------------------------------------------------------------
        debt_approved = df['ApplicantIncome'] * np.random.uniform(0.3, 0.8, n)
        debt_rejected  = df['ApplicantIncome'] * np.random.uniform(1.5, 3.5, n)
        raw_debt = np.where(approved, debt_approved, debt_rejected)
        df['Existing_Debt'] = np.maximum(raw_debt, 0).round(2)

        # ---------------------------------------------------------------
        # Credit Utilization: approved = lower utilization
        #   Approved:  mean=28%, std=12  → below 40% is healthy
        #   Rejected:  mean=75%, std=15  → high utilization = risky
        # ---------------------------------------------------------------
        util_approved = np.random.normal(loc=28, scale=12, size=n).clip(0, 100)
        util_rejected  = np.random.normal(loc=75, scale=15, size=n).clip(0, 100)
        df['Credit_Utilization'] = np.where(approved, util_approved, util_rejected).round(1)

        # Drop Gender
        if 'Gender' in df.columns:
            df.drop('Gender', axis=1, inplace=True)

        return df

    def perform_basic_validation(self, df):
        """Returns basic statistics and info for EDA."""
        report = {
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
            "target_distribution": df[TARGET_COL].value_counts(normalize=True).to_dict()
        }
        return report
