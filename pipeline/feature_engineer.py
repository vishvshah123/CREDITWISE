"""
CreditWise - Feature Engineer: Create domain-driven financial features.
Each feature has a financial justification for WHY it matters.
"""
import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.engineered_features = []
        self.reasoning_log = []
    
    def _log(self, name, formula, reason):
        self.reasoning_log.append({'feature': name, 'formula': formula, 'reasoning': reason})
        self.engineered_features.append(name)
    
    def engineer_features(self, df):
        df = df.copy()
        monthly_income = df['income'] / 12
        monthly_payment = df['loan_amount'] / df['loan_term_months']
        
        # 1. DTI - #1 ratio banks use
        df['debt_to_income_ratio'] = np.where(
            monthly_income > 0, (monthly_payment / monthly_income).clip(0, 5), 5.0)
        self._log('debt_to_income_ratio', 'monthly_payment/monthly_income',
                  'Banks reject DTI > 43%. Measures repayment burden.')
        
        # 2. LTV - collateral coverage
        df['loan_to_value_ratio'] = np.where(
            df['property_value'] > 0, (df['loan_amount'] / df['property_value']).clip(0, 3), 1.5)
        self._log('loan_to_value_ratio', 'loan_amount/property_value',
                  'LTV > 80% is risky. Measures collateral coverage.')
        
        # 3. Disposable income
        df['disposable_income'] = monthly_income - df['monthly_expenses'] - monthly_payment
        self._log('disposable_income', 'income - expenses - payment',
                  'Negative means applicant cannot afford the loan.')
        
        # 4. Savings-to-loan ratio
        df['savings_to_loan_ratio'] = np.where(
            df['loan_amount'] > 0, (df['savings_balance'] / df['loan_amount']).clip(0, 5), 0)
        self._log('savings_to_loan_ratio', 'savings/loan_amount',
                  'Financial cushion if income stops.')
        
        # 5. Income stability score
        emp_stab = df['employment_years'].clip(0, 30) / 30
        inc_norm = df['income'].clip(0) / (df['income'].max() if df['income'].max() > 0 else 1)
        df['income_stability_score'] = emp_stab * 0.6 + inc_norm * 0.4
        self._log('income_stability_score', '0.6*emp_years + 0.4*income_norm',
                  'Stable low income > unstable high income for repayment.')
        
        # 6. Credit utilization proxy
        df['credit_utilization'] = np.where(
            df['credit_score'] > 0,
            (df['existing_loans'] * 0.1 + (850 - df['credit_score']) / 550).clip(0, 1), 1.0)
        self._log('credit_utilization', 'loans*0.1 + (850-score)/550',
                  'High utilization (>30%) signals financial stress.')
        
        # 7. Risk score composite
        risk = (
            (1 - df['debt_to_income_ratio'].clip(0, 1)) * 0.25 +
            (1 - df['loan_to_value_ratio'].clip(0, 1)) * 0.15 +
            (df['credit_score'].clip(300, 850) - 300) / 550 * 0.30 +
            df['income_stability_score'] * 0.15 +
            (1 - df['previous_defaults'].clip(0, 3) / 3) * 0.15
        )
        df['risk_score_composite'] = risk.clip(0, 1)
        self._log('risk_score_composite', 'weighted: DTI+LTV+Credit+Stability+Defaults',
                  'Aggregated risk (0=worst, 1=best). Industry-standard weights.')
        
        # 8. Expense-to-income ratio
        df['expense_to_income_ratio'] = np.where(
            monthly_income > 0, (df['monthly_expenses'] / monthly_income).clip(0, 3), 3.0)
        self._log('expense_to_income_ratio', 'expenses/monthly_income',
                  'Ratio > 0.8 means applicant is already stretched thin.')
        
        return df
    
    def print_reasoning(self):
        print("\n" + "=" * 60)
        print("  FEATURE ENGINEERING REASONING")
        print("=" * 60)
        for i, e in enumerate(self.reasoning_log, 1):
            print(f"\n  {i}. {e['feature']}")
            print(f"     Formula: {e['formula']}")
            print(f"     WHY: {e['reasoning']}")
        print(f"\n  Total engineered features: {len(self.engineered_features)}")
        print("=" * 60)
