"""
CreditWise - Bias Detector: Detect biases in dataset and model predictions.
Uses Demographic Parity, Disparate Impact, and Equalized Odds.
"""
import pandas as pd
import numpy as np
from scipy import stats


class BiasDetector:
    """
    Detect bias on protected attributes (gender, age).
    
    WHY THIS MATTERS:
    ML models trained on historical data inherit past discrimination.
    If a bank historically approved fewer female loans, the model
    learns "female → reject" even if gender shouldn't matter.
    """
    
    def __init__(self, protected_attributes=None):
        self.protected_attributes = protected_attributes or ['gender']
        self.bias_report = {}
    
    def detect_data_bias(self, df, target_col='loan_approved'):
        """
        Detect bias in the RAW DATASET before any modeling.
        
        REASONING: We check bias at the data level FIRST because:
        1. If the data is biased, ANY model trained on it will be biased
        2. Data-level bias is easier to fix than model-level bias
        3. It tells us exactly WHERE the bias comes from
        """
        print("\n" + "=" * 60)
        print("  BIAS DETECTION — DATASET LEVEL")
        print("=" * 60)
        
        results = {}
        
        for attr in self.protected_attributes:
            if attr not in df.columns:
                continue
            
            print(f"\n  Protected Attribute: {attr.upper()}")
            print("  " + "-" * 40)
            
            groups = df.groupby(attr)[target_col]
            approval_rates = groups.mean()
            group_sizes = groups.count()
            
            # 1. Approval rates per group
            print(f"\n  Approval Rates:")
            for group, rate in approval_rates.items():
                size = group_sizes[group]
                print(f"    {group}: {rate*100:.1f}% (n={size:,})")
            
            # 2. Demographic Parity Difference
            # WHAT: Difference in approval rates between groups
            # IDEAL: 0 (equal rates). |DPD| > 0.05 = potential bias
            max_rate = approval_rates.max()
            min_rate = approval_rates.min()
            dpd = max_rate - min_rate
            
            print(f"\n  Demographic Parity Difference: {dpd:.4f}")
            print(f"    THRESHOLD: |DPD| < 0.05 = fair, > 0.10 = biased")
            if dpd > 0.10:
                print(f"    ⚠️  BIASED — {dpd:.1%} gap between groups")
            elif dpd > 0.05:
                print(f"    ⚠️  MARGINAL — {dpd:.1%} gap (borderline)")
            else:
                print(f"    ✅ FAIR — {dpd:.1%} gap is acceptable")
            
            # 3. Disparate Impact Ratio
            # WHAT: Ratio of approval rates (minority / majority)
            # IDEAL: 1.0. The "four-fifths rule": DIR < 0.8 = adverse impact
            majority_group = approval_rates.idxmax()
            minority_group = approval_rates.idxmin()
            dir_ratio = min_rate / max_rate if max_rate > 0 else 0
            
            print(f"\n  Disparate Impact Ratio: {dir_ratio:.4f}")
            print(f"    Advantaged group: {majority_group} ({max_rate:.1%})")
            print(f"    Disadvantaged group: {minority_group} ({min_rate:.1%})")
            print(f"    THRESHOLD: DIR >= 0.80 = fair (four-fifths rule)")
            if dir_ratio < 0.80:
                print(f"    ⚠️  FAILS four-fifths rule — adverse impact detected")
            else:
                print(f"    ✅ PASSES four-fifths rule")
            
            # 4. Chi-square test for statistical significance
            contingency = pd.crosstab(df[attr], df[target_col])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            
            print(f"\n  Chi-Square Test:")
            print(f"    Chi² = {chi2:.2f}, p-value = {p_value:.6f}")
            if p_value < 0.05:
                print(f"    ⚠️  Statistically significant bias (p < 0.05)")
            else:
                print(f"    ✅ No statistically significant bias")
            
            # 5. Feature correlation with protected attribute
            print(f"\n  Features correlated with {attr}:")
            numeric_df = df.select_dtypes(include=[np.number])
            if attr in numeric_df.columns:
                corrs = numeric_df.corr()[attr].drop([attr, target_col], errors='ignore')
                corrs = corrs.reindex(corrs.abs().sort_values(ascending=False).index)
                for feat, corr in list(corrs.items())[:5]:
                    flag = " ← PROXY RISK" if abs(corr) > 0.3 else ""
                    print(f"    {feat}: {corr:.3f}{flag}")
            
            results[attr] = {
                'approval_rates': approval_rates.to_dict(),
                'demographic_parity_diff': float(dpd),
                'disparate_impact_ratio': float(dir_ratio),
                'chi2': float(chi2),
                'p_value': float(p_value),
                'is_biased': dpd > 0.05 or dir_ratio < 0.80,
                'majority_group': majority_group,
                'minority_group': minority_group
            }
        
        self.bias_report['data_level'] = results
        print("\n" + "=" * 60)
        return results
    
    def detect_model_bias(self, y_true, y_pred, protected_values):
        """
        Detect bias in MODEL PREDICTIONS.
        
        REASONING: Even after data cleaning, the model may have learned
        biased patterns. We check if predictions are fair across groups.
        """
        print("\n" + "=" * 60)
        print("  BIAS DETECTION — MODEL PREDICTIONS")
        print("=" * 60)
        
        results = {}
        
        for attr_name, attr_values in protected_values.items():
            attr_name_clean = attr_name.replace('_test', '').replace('_train', '')
            print(f"\n  Protected Attribute: {attr_name_clean.upper()}")
            print("  " + "-" * 40)
            
            unique_groups = attr_values.unique()
            group_metrics = {}
            
            for group in unique_groups:
                mask = attr_values == group
                group_true = y_true[mask]
                group_pred = y_pred[mask]
                
                tp = ((group_pred == 1) & (group_true == 1)).sum()
                fp = ((group_pred == 1) & (group_true == 0)).sum()
                fn = ((group_pred == 0) & (group_true == 1)).sum()
                tn = ((group_pred == 0) & (group_true == 0)).sum()
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
                approval_rate = group_pred.mean()
                
                group_metrics[group] = {
                    'approval_rate': float(approval_rate),
                    'true_positive_rate': float(tpr),
                    'false_positive_rate': float(fpr),
                    'count': int(mask.sum())
                }
                
                print(f"\n    {group} (n={mask.sum():,}):")
                print(f"      Approval rate: {approval_rate:.1%}")
                print(f"      True Positive Rate: {tpr:.3f}")
                print(f"      False Positive Rate: {fpr:.3f}")
            
            # Equalized Odds check
            tprs = [m['true_positive_rate'] for m in group_metrics.values()]
            fprs = [m['false_positive_rate'] for m in group_metrics.values()]
            tpr_diff = max(tprs) - min(tprs)
            fpr_diff = max(fprs) - min(fprs)
            
            print(f"\n    Equalized Odds Check:")
            print(f"      TPR difference: {tpr_diff:.4f} (ideal: < 0.05)")
            print(f"      FPR difference: {fpr_diff:.4f} (ideal: < 0.05)")
            
            if tpr_diff > 0.05 or fpr_diff > 0.05:
                print(f"      ⚠️  Equalized odds VIOLATED")
            else:
                print(f"      ✅ Equalized odds satisfied")
            
            rates = [m['approval_rate'] for m in group_metrics.values()]
            dir_ratio = min(rates) / max(rates) if max(rates) > 0 else 0
            
            results[attr_name_clean] = {
                'group_metrics': group_metrics,
                'tpr_difference': float(tpr_diff),
                'fpr_difference': float(fpr_diff),
                'disparate_impact_ratio': float(dir_ratio),
                'equalized_odds_satisfied': tpr_diff <= 0.05 and fpr_diff <= 0.05
            }
        
        self.bias_report['model_level'] = results
        print("\n" + "=" * 60)
        return results
    
    def get_full_report(self):
        """Return the complete bias report."""
        return self.bias_report
