import pandas as pd
import numpy as np

class FeatureEngineer:
    def engineer_features(self, df):
        """Creates domain-specific financial features."""
        df = df.copy()
        
        # 1. Total Income
        if 'ApplicantIncome' in df.columns and 'CoapplicantIncome' in df.columns:
            df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
            
            # Apply log transformation to handle skewness
            df['Total_Income_Log'] = np.log1p(df['Total_Income'])
            df['ApplicantIncome_Log'] = np.log1p(df['ApplicantIncome'])
            df['LoanAmount_Log'] = np.log1p(df['LoanAmount'])
            
        # 2. EMI (Equated Monthly Installment approximation)
        if 'LoanAmount' in df.columns and 'Loan_Amount_Term' in df.columns:
            # LoanAmount is in thousands
            df['EMI'] = (df['LoanAmount'] * 1000) / df['Loan_Amount_Term']
            
        # 3. Balance Income (Income left after EMI)
        if 'Total_Income' in df.columns and 'EMI' in df.columns:
            df['Balance_Income'] = df['Total_Income'] - df['EMI']
            
        # 4. Debt-to-Income (DTI)
        if 'Existing_Debt' in df.columns and 'Total_Income' in df.columns and 'EMI' in df.columns:
            # Assuming Existing_Debt is total outstanding, we approximate monthly payment as 3% of existing debt
            monthly_existing_debt_payment = df['Existing_Debt'] * 0.03
            df['DTI'] = ((df['EMI'] + monthly_existing_debt_payment) / (df['Total_Income'] + 1)).round(3)
            
        # 5. EMI-to-Income Ratio
        if 'EMI' in df.columns and 'Total_Income' in df.columns:
            df['EMI_to_Income'] = (df['EMI'] / (df['Total_Income'] + 1)).round(3)
            
        # Drop original highly skewed features if log is created
        cols_to_drop = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Total_Income', 'Existing_Debt']
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
        
        return df
