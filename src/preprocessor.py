import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.config import NUMERICAL_COLS, CATEGORICAL_COLS, ID_COL

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputation_values = {}
        self.feature_cols = []
        self.cols_to_scale = []
        
    def handle_missing_values(self, df):
        """Impute missing values: Median for numerical, Mode for categorical."""
        df = df.copy()
        
        # Numerical
        for col in NUMERICAL_COLS:
            if col in df.columns and df[col].isnull().any():
                val = df[col].median()
                df[col] = df[col].fillna(val)
                self.imputation_values[col] = val
                
        # Categorical
        for col in CATEGORICAL_COLS:
            if col in df.columns and df[col].isnull().any():
                val = df[col].mode()[0]
                df[col] = df[col].fillna(val)
                self.imputation_values[col] = val
                
        return df

    def encode_categorical(self, df):
        """Encode categorical variables."""
        df = df.copy()
        
        # Label Encoding for Ordinal or Binary
        label_cols = ['Married', 'Education', 'Self_Employed']
        for col in label_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                    
        # Dependents (3+ to 3) and make numeric
        if 'Dependents' in df.columns:
            df['Dependents'] = df['Dependents'].astype(str).str.replace('3+', '3', regex=False).astype(int)
            
        # One Hot Encoding manually to avoid single-row inference issues
        if 'Property_Area' in df.columns:
            df['Property_Semiurban'] = (df['Property_Area'] == 'Semiurban').astype(int)
            df['Property_Urban'] = (df['Property_Area'] == 'Urban').astype(int)
            df.drop('Property_Area', axis=1, inplace=True)
            
        if 'Loan_Purpose' in df.columns:
            df['Purpose_Refinancing'] = (df['Loan_Purpose'] == 'Refinancing').astype(int)
            df['Purpose_Home_Improvement'] = (df['Loan_Purpose'] == 'Home Improvement').astype(int)
            df['Purpose_Personal'] = (df['Loan_Purpose'] == 'Personal').astype(int)
            df['Purpose_Education'] = (df['Loan_Purpose'] == 'Education').astype(int)
            df['Purpose_Business'] = (df['Loan_Purpose'] == 'Business').astype(int)
            df.drop('Loan_Purpose', axis=1, inplace=True)
            
        # Drop ID
        if ID_COL in df.columns:
            df.drop(ID_COL, axis=1, inplace=True)
            
        return df

    def scale_features(self, df, fit=True):
        """Standardize numerical columns."""
        df = df.copy()
        
        if fit:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self.cols_to_scale = [c for c in num_cols if df[c].nunique() > 2]
            df[self.cols_to_scale] = self.scaler.fit_transform(df[self.cols_to_scale])
        else:
            if self.cols_to_scale:
                df[self.cols_to_scale] = self.scaler.transform(df[self.cols_to_scale])
            
        self.feature_cols = df.columns.tolist()
        return df
