"""
Generates a realistic 3000-row synthetic loan dataset.
Follows US-style mortgage underwriting conventions.
ApplicantIncome = monthly salary (matching original dataset convention).
"""
import numpy as np
import pandas as pd
import os


def generate_dataset(n=3000, seed=42):
    np.random.seed(seed)

    # --- Monthly income (log-normal, median ~$5000/mo) ---
    monthly_income = np.random.lognormal(mean=np.log(5000), sigma=0.55, size=n).round(0)
    coapplicant_flag = np.random.choice([0, 1], size=n, p=[0.40, 0.60])
    co_income = (monthly_income * np.random.uniform(0.2, 0.7, n) * coapplicant_flag).round(0)
    total_monthly = monthly_income + co_income

    # --- Demographics ---
    married = np.where(coapplicant_flag == 1,
                       np.random.choice(['Yes', 'No'], size=n, p=[0.90, 0.10]),
                       np.random.choice(['Yes', 'No'], size=n, p=[0.20, 0.80]))
    dependents = np.random.choice(['0', '1', '2', '3+'], size=n, p=[0.40, 0.25, 0.20, 0.15])
    education  = np.random.choice(['Graduate', 'Not Graduate'], size=n, p=[0.78, 0.22])
    self_emp   = np.random.choice(['No', 'Yes'], size=n, p=[0.82, 0.18])
    prop_area  = np.random.choice(['Urban', 'Semiurban', 'Rural'], size=n, p=[0.38, 0.40, 0.22])

    # --- Loan details ---
    # Loan size: 20–50× monthly income (i.e., ~2–4 years income)
    loan_multiplier = np.random.uniform(18, 55, n)
    loan_amount = (monthly_income * loan_multiplier / 1000).round(0).clip(20, 700)  # in $K
    loan_term = np.random.choice([120, 180, 240, 300, 360], size=n, p=[0.05, 0.10, 0.15, 0.15, 0.55])

    # --- Credit profile (correlated with income) ---
    income_rank = (monthly_income - monthly_income.min()) / (monthly_income.max() - monthly_income.min())

    # Employment stability: higher income → more stable (realistic)
    employment_stability = (1.5 + 7 * income_rank + np.random.normal(0, 1.5, n)).clip(0, 35).round(1)

    # Existing MONTHLY debt obligations (car, credit cards, student loans)
    # Range: $200–$3000/month
    existing_monthly_debt = (200 + 2500 * (1 - income_rank) + np.random.normal(0, 300, n)).clip(0, 4000).round(0)

    # Credit utilization (higher income → lower utilization)
    credit_utilization = (75 - 45 * income_rank + np.random.normal(0, 12, n)).clip(0, 100).round(1)

    # Credit history: proxy for good payment record
    credit_history = np.where(credit_utilization < 50, 1.0, 0.0)

    # Existing debt total outstanding ≈ monthly obligation × 36 months
    existing_debt_total = (existing_monthly_debt * np.random.uniform(20, 50, n)).round(2)

    # --- Key ratios ---
    emi = (loan_amount * 1000) / loan_term  # Monthly loan payment
    dti = (emi + existing_monthly_debt) / (total_monthly + 1)
    income_coverage = total_monthly / (emi + 1)

    # ---------------------------------------------------------------
    # OUTCOME: Real underwriting criteria
    #   DTI < 43%  (Fannie Mae/Freddie Mac standard)
    #   Income covers EMI by at least 1.5×
    #   Credit utilization < 75%
    #   Employment > 6 months
    # ---------------------------------------------------------------
    approved = (
        (dti < 0.43) &
        (income_coverage > 1.5) &
        (credit_utilization < 75) &
        (employment_stability > 0.5)
    ).astype(int)

    # 5% realistic noise (exceptions, manual overrides)
    noise_mask = np.random.random(n) < 0.05
    approved = np.where(noise_mask, 1 - approved, approved)

    df = pd.DataFrame({
        'Married':              married,
        'Dependents':           dependents,
        'Education':            education,
        'Self_Employed':        self_emp,
        'ApplicantIncome':      monthly_income,
        'CoapplicantIncome':    co_income,
        'LoanAmount':           loan_amount,
        'Loan_Amount_Term':     loan_term,
        'Credit_History':       credit_history,
        'Property_Area':        prop_area,
        'Employment_Stability': employment_stability,
        'Existing_Debt':        existing_debt_total,
        'Credit_Utilization':   credit_utilization,
        'Loan_Status':          approved,
    })

    return df


if __name__ == '__main__':
    df = generate_dataset()
    out = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_loan_dataset.csv')
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} rows. Approval rate: {df['Loan_Status'].mean():.1%}")
