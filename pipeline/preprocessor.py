"""
╔══════════════════════════════════════════════════════════════════╗
║  CreditWise — Preprocessor                                       ║
║                                                                  ║
║  REASONING:                                                      ║
║  Raw data cannot be fed directly to ML models. We must:          ║
║  1. Handle missing values (models can't process NaN)            ║
║  2. Encode categorical features (models need numbers)           ║
║  3. Scale numeric features (gradient-based models need this)    ║
║  4. Detect outliers (extreme values can distort learning)       ║
║                                                                  ║
║  CRITICAL: Every imputation/transformation choice is documented ║
║  with WHY we chose that strategy for that specific feature.     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class Preprocessor:
    """
    Handle missing values, encode features, and prepare data for modeling.
    
    Design Decision: We separate preprocessing from feature engineering because:
    1. Preprocessing is about data QUALITY (fix what's broken)
    2. Feature engineering is about data ENRICHMENT (create new signals)
    3. Keeping them separate makes the pipeline auditable
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputation_values = {}
        self.numeric_cols = []
        self.categorical_cols = []
        self.feature_cols = []
        self.reasoning_log = []
    
    def _log_reasoning(self, step, feature, strategy, reason):
        """Log the reasoning behind each preprocessing decision."""
        self.reasoning_log.append({
            'step': step,
            'feature': feature,
            'strategy': strategy,
            'reason': reason
        })
    
    def handle_missing_values(self, df):
        """
        Impute missing values with feature-specific strategies.
        
        REASONING FOR EACH STRATEGY:
        ─────────────────────────────
        We DON'T use a one-size-fits-all approach because different features
        have different statistical properties and different reasons for being missing.
        """
        df = df.copy()
        
        # ── Income: Median imputation ──
        # WHY MEDIAN? Income is right-skewed (few very high earners pull the mean up).
        # Median is robust to this skewness and gives a more "typical" value.
        # WHY NOT MEAN? Mean = ~$65k, Median = ~$52k — mean would overestimate for most.
        if df['income'].isnull().any():
            median_income = df['income'].median()
            self.imputation_values['income'] = median_income
            df['income'].fillna(median_income, inplace=True)
            self._log_reasoning(
                'missing_values', 'income', f'Median ({median_income:,.0f})',
                'Income is right-skewed; median is robust to high-earner outliers'
            )
        
        # ── Credit Score: Median imputation + missing indicator ──
        # WHY MEDIAN? Credit scores are roughly normal; median ≈ mean here.
        # WHY MISSING INDICATOR? Missing credit score often means "thin file"
        # (young/new borrowers with no credit history) — this is itself a risk signal.
        if df['credit_score'].isnull().any():
            df['credit_score_missing'] = df['credit_score'].isnull().astype(int)
            median_credit = df['credit_score'].median()
            self.imputation_values['credit_score'] = median_credit
            df['credit_score'].fillna(median_credit, inplace=True)
            self._log_reasoning(
                'missing_values', 'credit_score',
                f'Median ({median_credit:.0f}) + missing indicator column',
                'Missing credit score signals "thin file" borrowers — '
                'the indicator captures this as a separate risk signal'
            )
        
        # ── Employment Years: Median imputation ──
        # WHY MEDIAN? Can't assume 0 (that means "just started") for missing data.
        # The person may have decades of experience but didn't fill the form.
        if df['employment_years'].isnull().any():
            median_emp = df['employment_years'].median()
            self.imputation_values['employment_years'] = median_emp
            df['employment_years'].fillna(median_emp, inplace=True)
            self._log_reasoning(
                'missing_values', 'employment_years', f'Median ({median_emp:.0f})',
                'Cannot assume 0 (unemployed) for missing; median is safest estimate'
            )
        
        # ── Savings Balance: Median + missing indicator ──
        # WHY MISSING INDICATOR? People who don't report savings may not HAVE savings.
        # This "missingness" pattern is a signal of financial behavior.
        if df['savings_balance'].isnull().any():
            df['savings_missing'] = df['savings_balance'].isnull().astype(int)
            median_savings = df['savings_balance'].median()
            self.imputation_values['savings_balance'] = median_savings
            df['savings_balance'].fillna(median_savings, inplace=True)
            self._log_reasoning(
                'missing_values', 'savings_balance',
                f'Median ({median_savings:,.0f}) + missing indicator',
                'Missing savings likely means no savings account — '
                'indicator captures this behavioral signal'
            )
        
        # ── Monthly Expenses: Median imputation ──
        # WHY MEDIAN? Expenses are skewed by high-income individuals.
        if df['monthly_expenses'].isnull().any():
            median_exp = df['monthly_expenses'].median()
            self.imputation_values['monthly_expenses'] = median_exp
            df['monthly_expenses'].fillna(median_exp, inplace=True)
            self._log_reasoning(
                'missing_values', 'monthly_expenses', f'Median ({median_exp:,.0f})',
                'Expenses are right-skewed; median avoids inflation from outliers'
            )
        
        # ── Property Value: Median imputation ──
        # WHY? Missing property value likely means unsecured loan (no collateral).
        # We use median to avoid penalizing unsecured loan applicants.
        if df['property_value'].isnull().any():
            df['property_missing'] = df['property_value'].isnull().astype(int)
            median_prop = df['property_value'].median()
            self.imputation_values['property_value'] = median_prop
            df['property_value'].fillna(median_prop, inplace=True)
            self._log_reasoning(
                'missing_values', 'property_value',
                f'Median ({median_prop:,.0f}) + missing indicator',
                'Missing = likely unsecured loan; indicator captures this distinction'
            )
        
        return df
    
    def encode_categorical(self, df, fit=True):
        """
        Encode categorical features to numeric.
        
        REASONING:
        - One-hot encoding for nominal features (no inherent order)
        - Label encoding for ordinal features (have natural order)
        - WHY NOT target encoding? Risk of data leakage in small datasets.
        """
        df = df.copy()
        
        # Ordinal: Education has a natural order
        # WHY ORDINAL? PhD > Master > Bachelor > High School for earning potential.
        education_order = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
        df['education_encoded'] = df['education'].map(education_order)
        self._log_reasoning(
            'encoding', 'education', 'Ordinal (0-3)',
            'Education has natural ordering that correlates with earning potential'
        )
        
        # Nominal: One-hot encode features without natural order
        nominal_cols = ['employment_type', 'marital_status', 'loan_purpose']
        for col in nominal_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                self._log_reasoning(
                    'encoding', col, 'One-hot (drop first)',
                    f'No natural order among {col} categories; drop_first avoids multicollinearity'
                )
        
        # Drop original categorical columns
        cols_to_drop = ['education', 'employment_type', 'marital_status', 'loan_purpose']
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
        
        return df
    
    def detect_outliers(self, df, columns=None):
        """
        Detect outliers using IQR method.
        
        REASONING: We use IQR (not z-score) because:
        1. IQR is robust to non-normal distributions (income, savings are skewed)
        2. z-score assumes normality — wrong for financial data
        3. We CAP outliers (winsorize) rather than removing them because:
           - Legitimate high-income applicants shouldn't be removed
           - Extreme values carry real information (very rich/poor applicants)
        """
        df = df.copy()
        if columns is None:
            columns = ['income', 'savings_balance', 'monthly_expenses', 
                       'loan_amount', 'property_value']
        
        outlier_report = {}
        for col in columns:
            if col not in df.columns:
                continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            outlier_report[col] = {
                'count': int(n_outliers),
                'percentage': round(n_outliers / len(df) * 100, 2),
                'lower_bound': round(lower, 2),
                'upper_bound': round(upper, 2)
            }
            
            # Winsorize: cap at bounds instead of removing
            df[col] = df[col].clip(lower=lower, upper=upper)
            
            self._log_reasoning(
                'outliers', col, f'Winsorized to [{lower:,.0f}, {upper:,.0f}]',
                f'{n_outliers} outliers capped (not removed) to preserve data volume'
            )
        
        return df, outlier_report
    
    def scale_features(self, df, target_col='loan_approved', fit=True):
        """
        Standardize numeric features.
        
        REASONING: StandardScaler (z-score normalization) because:
        1. Tree-based models (Random Forest, XGBoost) are scale-invariant, 
           but Logistic Regression is NOT — scaling is needed for fair comparison.
        2. StandardScaler preserves the shape of distributions.
        3. We fit ONLY on training data to prevent data leakage.
        """
        df = df.copy()
        
        # Identify numeric columns (excluding target and binary indicators)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = [target_col] + [c for c in numeric_cols if df[c].nunique() <= 2]
        scale_cols = [c for c in numeric_cols if c not in exclude]
        
        self.numeric_cols = scale_cols
        
        if fit:
            df[scale_cols] = self.scaler.fit_transform(df[scale_cols])
        else:
            df[scale_cols] = self.scaler.transform(df[scale_cols])
        
        self._log_reasoning(
            'scaling', f'{len(scale_cols)} features', 'StandardScaler (z-score)',
            'Needed for Logistic Regression; tree models are invariant but it doesn\'t hurt'
        )
        
        return df
    
    def prepare_data(self, df, target_col='loan_approved', test_size=0.2, random_state=42):
        """
        Full preprocessing pipeline: missing → encode → outliers → scale → split.
        
        REASONING for 80/20 split:
        - 80% training gives enough data for the model to learn patterns
        - 20% testing gives a reliable estimate of real-world performance
        - Stratified split ensures approval/rejection ratio is same in both sets
        """
        # Store gender column BEFORE encoding for bias analysis later
        gender_col = df['gender'].copy() if 'gender' in df.columns else None
        age_col = df['age'].copy() if 'age' in df.columns else None
        
        # Step 1: Handle missing values
        print("\n  📋 Step 1: Handling missing values...")
        df = self.handle_missing_values(df)
        
        # Step 2: Encode categorical features
        print("  📋 Step 2: Encoding categorical features...")
        df = self.encode_categorical(df)
        
        # Step 3: Remove protected attributes from features
        # REASONING: Gender should NOT be a feature the model uses.
        # Even after bias mitigation, keeping it risks the model
        # learning discriminatory patterns.
        protected_to_remove = ['gender']
        for col in protected_to_remove:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
                self._log_reasoning(
                    'feature_removal', col, 'Dropped',
                    'Protected attribute — must not be used for predictions to ensure fairness'
                )
        
        # Step 4: Detect and handle outliers
        print("  📋 Step 3: Detecting outliers...")
        df, outlier_report = self.detect_outliers(df)
        
        # Step 5: Split BEFORE scaling (prevent data leakage)
        print("  📋 Step 4: Train/test split (80/20 stratified)...")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        self.feature_cols = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Step 6: Scale features (fit on train only)
        print("  📋 Step 5: Scaling numeric features...")
        X_train = self.scale_features(X_train, target_col=None, fit=True)
        X_test = self.scale_features(X_test, target_col=None, fit=False)
        
        self._log_reasoning(
            'split', 'dataset', f'{1-test_size:.0%}/{test_size:.0%} stratified',
            f'Train: {len(X_train):,} samples, Test: {len(X_test):,} samples'
        )
        
        # Return gender/age info for bias analysis
        metadata = {}
        if gender_col is not None:
            metadata['gender_train'] = gender_col.iloc[X_train.index]
            metadata['gender_test'] = gender_col.iloc[X_test.index]
        if age_col is not None:
            metadata['age_train'] = age_col.iloc[X_train.index]
            metadata['age_test'] = age_col.iloc[X_test.index]
        
        return X_train, X_test, y_train, y_test, metadata
    
    def print_reasoning(self):
        """Print all preprocessing reasoning."""
        print("\n" + "=" * 60)
        print("  🧠 PREPROCESSING REASONING LOG")
        print("=" * 60)
        
        current_step = None
        for entry in self.reasoning_log:
            if entry['step'] != current_step:
                current_step = entry['step']
                print(f"\n  [{current_step.upper()}]")
            print(f"    • {entry['feature']}: {entry['strategy']}")
            print(f"      WHY: {entry['reason']}")
        
        print("\n" + "=" * 60)
