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
        
    def handle_missing_values(self, df):
        """Impute missing values: Median for numerical, Mode for categorical."""
        df = df.copy()
        
        # Numerical
        for col in NUMERICAL_COLS:
            if col in df.columns and df[col].isnull().any():
                val = df[col].median()
                df[col].fillna(val, inplace=True)
                self.imputation_values[col] = val
                
        # Categorical
        for col in CATEGORICAL_COLS:
            if col in df.columns and df[col].isnull().any():
                val = df[col].mode()[0]
                df[col].fillna(val, inplace=True)
                self.imputation_values[col] = val
                
        return df

    def encode_categorical(self, df):
        """Encode categorical variables."""
        df = df.copy()
        
        # Label Encoding for Ordinal or Binary
        label_cols = ['Gender', 'Married', 'Education', 'Self_Employed']
        for col in label_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                    
        # Dependents (3+ to 3) and make numeric
        if 'Dependents' in df.columns:
            df['Dependents'] = df['Dependents'].astype(str).str.replace('3+', '3').astype(int)
            
        # One Hot Encoding
        if 'Property_Area' in df.columns:
            dummies = pd.get_dummies(df['Property_Area'], prefix='Property', drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop('Property_Area', axis=1, inplace=True)
            
        # Drop ID
        if ID_COL in df.columns:
            df.drop(ID_COL, axis=1, inplace=True)
            
        return df

    def scale_features(self, df, fit=True):
        """Standardize numerical columns."""
        df = df.copy()
        
        # All columns except target and binary categorical
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_scale = [c for c in num_cols if df[c].nunique() > 2]
        
        if fit:
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        else:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
            
        self.feature_cols = df.columns.tolist()
        return df
