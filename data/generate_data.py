"""
╔══════════════════════════════════════════════════════════════════╗
║  CreditWise — Synthetic Loan Dataset Generator                   ║
║                                                                  ║
║  WHY SYNTHETIC DATA?                                             ║
║  1. We can inject KNOWN biases (gender, age) so our bias         ║
║     detector has ground truth to validate against.               ║
║  2. We control the true data-generating process, so we know      ║
║     exactly which features drive approval.                       ║
║  3. No privacy/licensing concerns with real financial data.      ║
║  4. We can ensure realistic distributions that mirror actual     ║
║     lending data (right-skewed income, bounded credit scores).   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import os

def generate_loan_dataset(n_samples=5000, random_state=42, bias_strength=0.15):
    """
    Generate a realistic synthetic loan application dataset with INTENTIONAL biases.
    
    REASONING FOR EACH FEATURE:
    ---------------------------
    Every feature is chosen because it mirrors what real banks use in underwriting.
    We deliberately inject gender and age biases so our bias detector can find them.
    
    Parameters
    ----------
    n_samples : int
        Number of loan applications to generate.
    random_state : int
        Random seed for reproducibility.
    bias_strength : float
        How much bias to inject (0 = none, 0.3 = strong). Default 0.15 = moderate.
        
    Returns
    -------
    pd.DataFrame
        Complete loan dataset with target column 'loan_approved'.
    """
    np.random.seed(random_state)
    
    # ──────────────────────────────────────────────────────────────
    # STEP 1: Generate demographic features
    # REASONING: These are standard applicant attributes collected
    #            during any loan application process.
    # ──────────────────────────────────────────────────────────────
    
    # Age: Normal distribution centered at 35, clipped to 21-65
    # WHY: Working-age adults are typical loan applicants
    age = np.clip(np.random.normal(35, 10, n_samples), 21, 65).astype(int)
    
    # Gender: Binary for simplicity (Male/Female)
    # WHY: This is a PROTECTED ATTRIBUTE — we inject bias here to test fairness
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45])
    
    # Education: Ordinal levels
    # WHY: Education correlates with earning potential and job stability
    education = np.random.choice(
        ['High School', 'Bachelor', 'Master', 'PhD'], 
        n_samples,
        p=[0.25, 0.40, 0.25, 0.10]
    )
    
    # Marital status
    # WHY: Affects household income calculations and risk assessment
    marital_status = np.random.choice(
        ['Single', 'Married', 'Divorced'], 
        n_samples,
        p=[0.35, 0.50, 0.15]
    )
    
    # Number of dependents: Poisson distribution
    # WHY: More dependents = higher monthly expenses = harder to repay
    num_dependents = np.random.poisson(1.2, n_samples).clip(0, 5)
    
    # ──────────────────────────────────────────────────────────────
    # STEP 2: Generate financial features
    # REASONING: These directly measure ability to repay.
    # ──────────────────────────────────────────────────────────────
    
    # Income: Log-normal distribution (right-skewed, as real income data is)
    # WHY: Primary ability-to-repay indicator. Log-normal captures the
    #      reality that most people earn moderate amounts with a long right tail.
    base_income = np.random.lognormal(mean=10.8, sigma=0.5, size=n_samples)
    
    # BIAS INJECTION: Reduce female income by bias_strength factor
    # This simulates the real-world gender pay gap that models can learn
    income_bias = np.where(gender == 'Female', 1 - bias_strength, 1.0)
    income = (base_income * income_bias).astype(int)
    
    # Employment type
    # WHY: Salaried employees have more predictable income than self-employed
    employment_type = np.random.choice(
        ['Salaried', 'Self-Employed', 'Freelancer', 'Business Owner'],
        n_samples,
        p=[0.45, 0.25, 0.15, 0.15]
    )
    
    # Employment years: Correlated with age (older = more experience)
    # WHY: Job stability = repayment reliability. Longer tenure = lower risk.
    employment_years = np.clip(
        (age - 21) * np.random.uniform(0.1, 0.7, n_samples) + np.random.normal(0, 2, n_samples),
        0, 40
    ).astype(int)
    
    # Credit score: Normal distribution bounded 300-850 (US FICO range)
    # WHY: Direct creditworthiness measure. THE single most important feature
    #      in real lending decisions.
    credit_score = np.clip(
        np.random.normal(680, 80, n_samples), 300, 850
    ).astype(int)
    
    # Existing loans count
    # WHY: More existing debt = higher risk of over-leverage
    existing_loans = np.random.poisson(1.5, n_samples).clip(0, 8)
    
    # Monthly expenses: Correlated with income and dependents
    # WHY: High expenses relative to income reduces repayment capacity
    monthly_expenses = (
        income * np.random.uniform(0.3, 0.6, n_samples) / 12 +
        num_dependents * np.random.uniform(200, 500, n_samples)
    ).astype(int)
    
    # Savings balance: Log-normal, correlated with income
    # WHY: Financial cushion for emergencies; shows financial discipline
    savings_balance = (
        np.random.lognormal(mean=8, sigma=1.5, size=n_samples) *
        (income / income.mean())
    ).astype(int)
    
    # Previous defaults: Critical risk indicator
    # WHY: Past defaults are THE strongest predictor of future defaults.
    #      This is heavily weighted in real credit scoring.
    previous_defaults = np.random.choice(
        [0, 1, 2, 3], n_samples, p=[0.70, 0.18, 0.08, 0.04]
    )
    
    # Has co-applicant
    # WHY: Co-applicants reduce lender risk by providing additional income guarantee
    has_co_applicant = np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
    
    # ──────────────────────────────────────────────────────────────
    # STEP 3: Generate loan-specific features
    # REASONING: These describe the loan being requested.
    # ──────────────────────────────────────────────────────────────
    
    # Loan amount: Related to income (people request loans proportional to income)
    # WHY: Larger loans = higher default risk relative to income
    loan_amount = (
        income * np.random.uniform(0.5, 3.0, n_samples)
    ).astype(int)
    
    # Loan term: Standard mortgage/personal loan terms
    # WHY: Longer terms = more uncertainty and total interest cost
    loan_term_months = np.random.choice(
        [12, 24, 36, 48, 60, 120, 180, 240, 360],
        n_samples,
        p=[0.05, 0.10, 0.15, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10]
    )
    
    # Property value (for secured loans): Related to loan amount
    # WHY: Collateral coverage — if borrower defaults, bank can recover via property
    property_value = (
        loan_amount * np.random.uniform(1.0, 2.5, n_samples)
    ).astype(int)
    
    # Loan purpose
    # WHY: Different purposes have different risk profiles
    loan_purpose = np.random.choice(
        ['Home', 'Car', 'Education', 'Personal', 'Business'],
        n_samples,
        p=[0.30, 0.20, 0.15, 0.20, 0.15]
    )
    
    # ──────────────────────────────────────────────────────────────
    # STEP 4: Generate target variable (loan_approved)
    # REASONING: The approval decision is based on realistic financial
    #            rules PLUS intentional bias, so our model learns both
    #            legitimate patterns AND biased patterns.
    # ──────────────────────────────────────────────────────────────
    
    # Calculate a "true risk score" based on legitimate financial factors
    risk_score = np.zeros(n_samples)
    
    # Credit score contribution (strongest factor, as in reality)
    risk_score += (credit_score - 300) / 550 * 35  # 0-35 points
    
    # Income contribution
    income_percentile = np.clip((income - income.min()) / (income.max() - income.min()), 0, 1)
    risk_score += income_percentile * 20  # 0-20 points
    
    # Debt-to-income proxy
    monthly_income = income / 12
    estimated_monthly_payment = loan_amount / loan_term_months
    dti = np.where(monthly_income > 0, estimated_monthly_payment / monthly_income, 1)
    risk_score += np.clip(1 - dti, 0, 1) * 15  # 0-15 points
    
    # Previous defaults (heavily penalized)
    risk_score -= previous_defaults * 10  # -0 to -30 points
    
    # Employment stability
    risk_score += np.clip(employment_years / 20, 0, 1) * 10  # 0-10 points
    
    # Savings buffer
    savings_ratio = np.where(loan_amount > 0, savings_balance / loan_amount, 0)
    risk_score += np.clip(savings_ratio, 0, 1) * 10  # 0-10 points
    
    # Co-applicant bonus
    risk_score += has_co_applicant * 5  # 0-5 points
    
    # Property value coverage
    ltv = np.where(property_value > 0, loan_amount / property_value, 1)
    risk_score += np.clip(1 - ltv, 0, 1) * 5  # 0-5 points
    
    # ──────────────────────────────────────────────────────────────
    # BIAS INJECTION INTO TARGET
    # REASONING: We deliberately make it harder for females and
    #            younger applicants to get approved. This simulates
    #            historical biases in lending data that models can
    #            learn and perpetuate.
    # ──────────────────────────────────────────────────────────────
    
    # Gender bias: Reduce risk score for females
    gender_penalty = np.where(gender == 'Female', -bias_strength * 15, 0)
    risk_score += gender_penalty
    
    # Age bias: Penalize younger applicants (under 30) more than warranted
    age_penalty = np.where(age < 30, -bias_strength * 10, 0)
    risk_score += age_penalty
    
    # Add noise to make it realistic (not perfectly separable)
    risk_score += np.random.normal(0, 5, n_samples)
    
    # Normalize to 0-100
    risk_score = np.clip(risk_score, 0, 100)
    
    # Threshold: approve if risk score > 45 (roughly 60-65% approval rate)
    loan_approved = (risk_score > 45).astype(int)
    
    # ──────────────────────────────────────────────────────────────
    # STEP 5: Introduce realistic missing values
    # REASONING: Real data always has missing values. We introduce
    #            them in patterns that mirror reality:
    #            - Self-employed people less likely to report income
    #            - Younger people may not have credit history
    #            - Savings data often incomplete
    # ──────────────────────────────────────────────────────────────
    
    # Create DataFrame first
    df = pd.DataFrame({
        'age': age,
        'gender': gender,
        'education': education,
        'marital_status': marital_status,
        'num_dependents': num_dependents,
        'income': income.astype(float),
        'employment_type': employment_type,
        'employment_years': employment_years.astype(float),
        'credit_score': credit_score.astype(float),
        'existing_loans': existing_loans,
        'monthly_expenses': monthly_expenses.astype(float),
        'savings_balance': savings_balance.astype(float),
        'previous_defaults': previous_defaults,
        'has_co_applicant': has_co_applicant,
        'loan_amount': loan_amount.astype(float),
        'loan_term_months': loan_term_months,
        'property_value': property_value.astype(float),
        'loan_purpose': loan_purpose,
        'loan_approved': loan_approved
    })
    
    # Inject missing values (5-12% per feature, with patterns)
    n = len(df)
    
    # Income: 8% missing, more likely for self-employed
    income_mask = np.random.random(n) < np.where(
        df['employment_type'].isin(['Self-Employed', 'Freelancer']), 0.15, 0.05
    )
    df.loc[income_mask, 'income'] = np.nan
    
    # Credit score: 6% missing, more likely for young applicants (thin file)
    credit_mask = np.random.random(n) < np.where(df['age'] < 25, 0.20, 0.04)
    df.loc[credit_mask, 'credit_score'] = np.nan
    
    # Employment years: 7% missing
    emp_mask = np.random.random(n) < 0.07
    df.loc[emp_mask, 'employment_years'] = np.nan
    
    # Savings balance: 10% missing (many people don't disclose)
    savings_mask = np.random.random(n) < 0.10
    df.loc[savings_mask, 'savings_balance'] = np.nan
    
    # Monthly expenses: 5% missing
    expense_mask = np.random.random(n) < 0.05
    df.loc[expense_mask, 'monthly_expenses'] = np.nan
    
    # Property value: 8% missing (unsecured loans don't have property)
    prop_mask = np.random.random(n) < 0.08
    df.loc[prop_mask, 'property_value'] = np.nan
    
    return df


def save_dataset(df, output_dir='data'):
    """Save the dataset to CSV with a summary."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'loan_applications.csv')
    df.to_csv(filepath, index=False)
    
    print("=" * 60)
    print("  CreditWise — Dataset Generation Summary")
    print("=" * 60)
    print(f"\n  Total applications: {len(df):,}")
    print(f"  Approved: {df['loan_approved'].sum():,} ({df['loan_approved'].mean()*100:.1f}%)")
    print(f"  Rejected: {(1 - df['loan_approved']).sum():,} ({(1-df['loan_approved'].mean())*100:.1f}%)")
    print(f"\n  Features: {len(df.columns) - 1}")
    print(f"  Missing values: {df.isnull().sum().sum():,} total")
    print(f"\n  Gender distribution:")
    for g in df['gender'].unique():
        subset = df[df['gender'] == g]
        print(f"    {g}: {len(subset):,} ({subset['loan_approved'].mean()*100:.1f}% approval rate)")
    print(f"\n  Saved to: {filepath}")
    print("=" * 60)
    
    return filepath


if __name__ == '__main__':
    df = generate_loan_dataset()
    save_dataset(df)
