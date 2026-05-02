import pandas as pd
import numpy as np

class BiasMitigator:
    def __init__(self, protected_attribute='Gender'):
        self.protected_attribute = protected_attribute

    def detect_bias(self, y_true, y_pred, protected_series):
        """Measures Demographic Parity Difference and Equal Opportunity Difference."""
        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'protected': protected_series
        })
        
        # Demographic Parity: P(Y_hat=1 | group=A) - P(Y_hat=1 | group=B)
        rates = df.groupby('protected')['y_pred'].mean()
        if len(rates) == 2:
            dpd = abs(rates.iloc[0] - rates.iloc[1])
        else:
            dpd = 0
            
        # Equal Opportunity: TPR_A - TPR_B
        tpr = {}
        for group in df['protected'].unique():
            group_df = df[(df['protected'] == group) & (df['y_true'] == 1)]
            if len(group_df) > 0:
                tpr[group] = group_df['y_pred'].mean()
            else:
                tpr[group] = 0
                
        if len(tpr) == 2:
            vals = list(tpr.values())
            eod = abs(vals[0] - vals[1])
        else:
            eod = 0
            
        return {
            'demographic_parity_diff': dpd,
            'equal_opportunity_diff': eod
        }

    def compute_reweighting(self, df, target_col):
        """Computes sample weights to equalize the outcome probabilities across groups."""
        weights = np.ones(len(df))
        
        if self.protected_attribute not in df.columns:
            return weights
            
        protected_col = df[self.protected_attribute]
        target = df[target_col]
        
        for g in protected_col.unique():
            for c in target.unique():
                mask = (protected_col == g) & (target == c)
                n_gc = mask.sum()
                if n_gc == 0:
                    continue
                
                n_g = (protected_col == g).sum()
                n_c = (target == c).sum()
                n_total = len(df)
                
                # Weight = (P(group) * P(class)) / P(group AND class)
                w = (n_g * n_c) / (n_total * n_gc)
                weights[mask] = w
                
        return weights
