import pandas as pd
import numpy as np
from src.config import DATA_PATH, TARGET_COL

class DataLoader:
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        
    def load_data(self):
        """Loads the raw dataset and maps target to binary."""
        df = pd.read_csv(self.data_path)

        # Map target variable FIRST so we can condition on it
        df[TARGET_COL] = df[TARGET_COL].map({'Y': 1, 'N': 0})

        np.random.seed(42)
        n = len(df)
        approved = df[TARGET_COL].values.astype(float)  # 1 = approved, 0 = rejected

        # ---------------------------------------------------------------
        # LATENT FINANCIAL HEALTH SCORE
        # A single underlying [-3, 3] variable representing true creditworthiness.
        # All synthetic features are derived from it — making them all naturally
        # intercorrelated, just like real banking data.
        #
        # Sources of health:
        #   50% → loan outcome (the ground truth signal)
        #   30% → applicant income (normalized, already in dataset)
        #   10% → whether married (slight positive signal)
        #   10% → pure noise (individual variation)
        # ---------------------------------------------------------------
        income_z = (df['ApplicantIncome'] - df['ApplicantIncome'].mean()) / (df['ApplicantIncome'].std() + 1)
        income_z = income_z.clip(-2, 2).values

        married_signal = (df['Married'] == 'Yes').astype(float).values * 0.3

        outcome_signal = (approved * 2 - 1)  # +1 approved, -1 rejected

        health = (
            0.50 * outcome_signal +
            0.30 * income_z +
            0.10 * married_signal +
            0.10 * np.random.normal(0, 1, n)
        )
        # Normalize to roughly [-2, 2]
        health = health.clip(-2.5, 2.5)

        # ---------------------------------------------------------------
        # EMPLOYMENT STABILITY (yrs)  ← strong positive link to health
        #   High health  → 8–12 yrs (senior, stable)
        #   Low health   → 0–3 yrs  (recent hires, unstable)
        # ---------------------------------------------------------------
        emp_base = 5 + 3 * health  # ranges ~[-1, 11] before noise
        emp_noise = np.random.normal(0, 1.2, n)
        df['Employment_Stability'] = (emp_base + emp_noise).clip(0, 30).round(1)

        # ---------------------------------------------------------------
        # EXISTING DEBT ($)  ← directly driven by HEALTH (not just income)
        #   High health  → low debt (0.2–0.5× income)
        #   Low health   → high debt (2.0–4.0× income)
        #   Also: higher income naturally means more credit available so
        #         we add a small positive income nudge on top.
        # ---------------------------------------------------------------
        # debt_multiplier: at health=+2 → 0.15, at health=-2 → 2.55
        debt_multiplier = 1.20 - 0.55 * health
        debt_noise = np.random.normal(0, 0.10, n)
        debt_multiplier = (debt_multiplier + debt_noise).clip(0.05, 5.0)
        # Base: income × multiplier (lower health → bigger multiple of income owed)
        # Income boost: higher earners can service more debt → add small fraction
        income_boost = df['ApplicantIncome'].values * 0.05
        df['Existing_Debt'] = (df['ApplicantIncome'].values * debt_multiplier + income_boost).round(2)

        # ---------------------------------------------------------------
        # CREDIT UTILIZATION (%) ← driven by HEALTH + Existing_Debt/Income ratio
        #   High debt-to-income → high utilization
        #   Formula: base from health, nudged upward by actual debt ratio
        # ---------------------------------------------------------------
        debt_to_income = (df['Existing_Debt'].values / (df['ApplicantIncome'].values + 1)).clip(0, 5)
        util_center = 30 - 15 * health + 10 * debt_to_income.clip(0, 3)
        util_noise = np.random.normal(0, 6, n)
        df['Credit_Utilization'] = (util_center + util_noise).clip(0, 100).round(1)

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
