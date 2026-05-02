"""
╔══════════════════════════════════════════════════════════════════╗
║  CreditWise — Data Loader & Initial Inspection                   ║
║                                                                  ║
║  REASONING:                                                      ║
║  Before ANY modeling, we must understand:                        ║
║  1. What data do we have? (shape, types, features)              ║
║  2. How much is missing? (impacts imputation strategy)          ║
║  3. What are the distributions? (impacts scaling choices)       ║
║  4. Is the target balanced? (impacts sampling strategy)         ║
║  5. Are there obvious correlations? (impacts feature selection) ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np


class DataLoader:
    """
    Load and inspect the loan application dataset.
    
    WHY A CLASS?
    We encapsulate data loading so the inspection results (stats, missing info)
    are stored alongside the data for downstream pipeline stages to reference.
    """
    
    def __init__(self):
        self.df = None
        self.inspection_report = {}
        
    def load(self, filepath):
        """
        Load CSV data and perform initial inspection.
        
        REASONING: We read the CSV and immediately compute summary statistics
        so any data quality issues are caught before expensive processing.
        """
        self.df = pd.read_csv(filepath)
        self._inspect()
        return self.df
    
    def load_from_dataframe(self, df):
        """Load from an existing DataFrame (e.g., from the generator)."""
        self.df = df.copy()
        self._inspect()
        return self.df
    
    def _inspect(self):
        """
        Comprehensive data inspection.
        
        REASONING for each check:
        - Shape: Know the scale of data we're working with
        - Types: Identify which features are numeric vs categorical
        - Missing: Decide imputation strategy per feature
        - Target balance: Decide if we need SMOTE or class weights
        - Correlations: Spot redundant features and potential proxies
        """
        df = self.df
        
        # Basic shape
        self.inspection_report['shape'] = df.shape
        self.inspection_report['dtypes'] = df.dtypes.to_dict()
        
        # Missing value analysis
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        self.inspection_report['missing'] = pd.DataFrame({
            'count': missing,
            'percentage': missing_pct
        }).sort_values('count', ascending=False)
        
        # Numeric feature stats
        self.inspection_report['numeric_stats'] = df.describe()
        
        # Target balance
        if 'loan_approved' in df.columns:
            target_counts = df['loan_approved'].value_counts()
            self.inspection_report['target_balance'] = {
                'approved': int(target_counts.get(1, 0)),
                'rejected': int(target_counts.get(0, 0)),
                'approval_rate': float(df['loan_approved'].mean()),
                'is_imbalanced': abs(df['loan_approved'].mean() - 0.5) > 0.15
            }
        
        # Categorical feature cardinality
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.inspection_report['categorical_info'] = {
            col: {
                'unique_values': df[col].nunique(),
                'top_values': df[col].value_counts().head(5).to_dict()
            }
            for col in cat_cols
        }
        
        # Numeric correlations with target
        if 'loan_approved' in df.columns:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'loan_approved' in num_cols:
                correlations = df[num_cols].corr()['loan_approved'].drop('loan_approved')
                self.inspection_report['target_correlations'] = correlations.sort_values(
                    key=abs, ascending=False
                ).to_dict()
    
    def print_report(self):
        """Print a formatted inspection report."""
        report = self.inspection_report
        
        print("\n" + "=" * 60)
        print("  📊 DATA INSPECTION REPORT")
        print("=" * 60)
        
        print(f"\n  Shape: {report['shape'][0]:,} rows × {report['shape'][1]} columns")
        
        # Missing values
        missing = report['missing']
        missing_cols = missing[missing['count'] > 0]
        if len(missing_cols) > 0:
            print(f"\n  ⚠️  Missing Values ({len(missing_cols)} features affected):")
            for col, row in missing_cols.iterrows():
                print(f"    • {col}: {int(row['count']):,} ({row['percentage']}%)")
        else:
            print("\n  ✅ No missing values")
        
        # Target balance
        if 'target_balance' in report:
            tb = report['target_balance']
            print(f"\n  🎯 Target Distribution:")
            print(f"    Approved: {tb['approved']:,} ({tb['approval_rate']*100:.1f}%)")
            print(f"    Rejected: {tb['rejected']:,} ({(1-tb['approval_rate'])*100:.1f}%)")
            if tb['is_imbalanced']:
                print(f"    ⚠️  Dataset is IMBALANCED — will use class weights")
            else:
                print(f"    ✅ Dataset is reasonably balanced")
        
        # Top correlations with target
        if 'target_correlations' in report:
            print(f"\n  📈 Top Feature Correlations with Approval:")
            corrs = report['target_correlations']
            for feat, corr in list(corrs.items())[:5]:
                direction = "↑ positive" if corr > 0 else "↓ negative"
                print(f"    • {feat}: {corr:.3f} ({direction})")
        
        print("\n" + "=" * 60)
        
        return report
