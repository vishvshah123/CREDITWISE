"""
CreditWise - Bias Reducer: Neutralize detected biases using multiple strategies.

Pipeline: Detect → Quantify → Neutralize → Verify
Strategies: Reweighting, Proxy Removal, Threshold Adjustment
"""
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight


class BiasReducer:
    """
    Reduce bias through pre-processing and in-processing techniques.
    
    WHY MULTIPLE STRATEGIES?
    No single technique removes all bias. We layer them:
    1. Pre-processing: Fix the DATA (remove proxies, reweight samples)
    2. In-processing: Fix the TRAINING (sample weights, balanced classes)
    3. Post-processing: Fix the OUTPUT (group-specific thresholds)
    """
    
    def __init__(self):
        self.reduction_log = []
        self.sample_weights = None
        self.proxy_features = []
        self.group_thresholds = {}
    
    def _log(self, strategy, action, impact):
        self.reduction_log.append({
            'strategy': strategy, 'action': action, 'impact': impact
        })
    
    def remove_proxy_features(self, df, protected_col, threshold=0.25):
        """
        Identify and remove features that are proxies for protected attributes.
        
        REASONING: Even if we remove 'gender' from features, other features
        like 'income' may be highly correlated with gender (gender pay gap).
        These proxies allow the model to indirectly discriminate.
        
        WHY threshold=0.25?
        - |corr| > 0.3 is a well-known "moderate correlation" cutoff
        - We use 0.25 to be conservative (catch weaker proxies too)
        - But not too low (< 0.1) which would remove too many useful features
        """
        print("\n  📋 Bias Reduction Step 1: Proxy Feature Detection")
        print("  " + "-" * 50)
        
        if protected_col not in df.columns:
            print(f"    {protected_col} not in dataframe, skipping proxy detection")
            return df, []
        
        # Encode protected attribute if categorical
        if df[protected_col].dtype == 'object':
            protected_encoded = pd.get_dummies(df[protected_col], drop_first=True)
            protected_series = protected_encoded.iloc[:, 0]
        else:
            protected_series = df[protected_col]
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = [protected_col, 'loan_approved']
        check_cols = [c for c in numeric_cols if c not in exclude]
        
        proxies_found = []
        for col in check_cols:
            corr = abs(df[col].corr(protected_series))
            if corr > threshold:
                proxies_found.append((col, corr))
                print(f"    ⚠️  {col}: correlation = {corr:.3f} (PROXY — above {threshold} threshold)")
        
        if not proxies_found:
            print(f"    ✅ No proxy features found (threshold: {threshold})")
        else:
            print(f"\n    Found {len(proxies_found)} proxy features.")
            print(f"    DECISION: We do NOT remove proxies outright because they may")
            print(f"    carry legitimate predictive signal. Instead, we use reweighting")
            print(f"    to reduce their discriminatory impact.")
        
        self.proxy_features = [p[0] for p in proxies_found]
        self._log('proxy_detection', f'Found {len(proxies_found)} proxies', 
                  'Will mitigate via reweighting instead of removal')
        
        return df, proxies_found
    
    def compute_reweighting(self, df, protected_col, target_col='loan_approved'):
        """
        Compute sample weights to equalize outcome distribution across groups.
        
        REASONING: If females have 55% approval and males have 70%, we need to
        UPWEIGHT female-approved and male-rejected samples so the model sees
        a balanced view across groups. This is the safest bias reduction method
        because it doesn't change the data, only the importance of each sample.
        
        HOW IT WORKS:
        weight(group, outcome) = P(outcome) / P(outcome | group)
        This makes the model "see" equal approval rates across groups.
        """
        print("\n  📋 Bias Reduction Step 2: Sample Reweighting")
        print("  " + "-" * 50)
        
        if protected_col not in df.columns:
            print(f"    {protected_col} not found, using uniform weights")
            self.sample_weights = np.ones(len(df))
            return self.sample_weights
        
        # Calculate expected vs observed frequencies
        overall_rate = df[target_col].mean()
        groups = df[protected_col].unique()
        
        weights = np.ones(len(df))
        
        for group in groups:
            for outcome in [0, 1]:
                mask = (df[protected_col] == group) & (df[target_col] == outcome)
                
                # Expected proportion (if no bias)
                p_outcome = (df[target_col] == outcome).mean()
                # Observed proportion in this group
                p_outcome_given_group = (df.loc[df[protected_col] == group, target_col] == outcome).mean()
                
                if p_outcome_given_group > 0:
                    w = p_outcome / p_outcome_given_group
                    weights[mask] = w
                    
                    n = mask.sum()
                    print(f"    {group} × {'Approved' if outcome else 'Rejected'}: "
                          f"weight = {w:.3f} (n={n:,})")
        
        # Normalize weights to mean = 1
        weights = weights / weights.mean()
        self.sample_weights = weights
        
        self._log('reweighting', f'Computed weights for {len(groups)} groups',
                  f'Weights range: [{weights.min():.3f}, {weights.max():.3f}]')
        
        print(f"\n    ✅ Sample weights computed (mean=1.0, range=[{weights.min():.3f}, {weights.max():.3f}])")
        print(f"    These will be passed to model training via sample_weight parameter.")
        
        return weights
    
    def get_balanced_class_weights(self, y):
        """
        Compute class weights for imbalanced target.
        
        REASONING: If 65% approved and 35% rejected, the model will
        be biased toward predicting "approved" because it minimizes
        overall error. Class weights fix this by making each class
        equally important regardless of frequency.
        """
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        weight_dict = dict(zip(classes, class_weights))
        
        print(f"\n  📋 Class Weights (for imbalanced target):")
        for cls, w in weight_dict.items():
            label = 'Approved' if cls == 1 else 'Rejected'
            print(f"    {label}: {w:.3f}")
        
        return weight_dict
    
    def compute_group_thresholds(self, y_true, y_proba, protected_values, base_threshold=0.5):
        """
        Post-processing: Adjust decision thresholds per group to achieve fairness.
        
        REASONING: If the model still shows bias after reweighting, we can
        adjust the approval threshold for each group independently.
        Example: If females need 0.6 probability but males only need 0.4,
        we lower the female threshold to equalize approval rates.
        
        This is a LAST RESORT because it treats groups differently,
        which may raise legal concerns. We prefer fixing the model directly.
        """
        print("\n  📋 Bias Reduction Step 3: Group-Specific Thresholds")
        print("  " + "-" * 50)
        
        overall_approval_rate = y_true.mean()
        
        for attr_name, attr_values in protected_values.items():
            groups = attr_values.unique()
            thresholds = {}
            
            for group in groups:
                mask = attr_values == group
                group_proba = y_proba[mask]
                
                # Find threshold that gives this group the overall approval rate
                # Binary search for optimal threshold
                low, high = 0.0, 1.0
                for _ in range(100):
                    mid = (low + high) / 2
                    predicted_rate = (group_proba >= mid).mean()
                    if predicted_rate > overall_approval_rate:
                        low = mid
                    else:
                        high = mid
                
                thresholds[group] = round(mid, 4)
                predicted_rate = (group_proba >= mid).mean()
                print(f"    {group}: threshold = {mid:.4f} → approval rate = {predicted_rate:.1%}")
            
            self.group_thresholds[attr_name] = thresholds
        
        print(f"\n    Base threshold: {base_threshold}")
        print(f"    ⚠️  Group thresholds are a LAST RESORT — prefer reweighting first")
        
        return self.group_thresholds
    
    def print_summary(self):
        """Print bias reduction summary."""
        print("\n" + "=" * 60)
        print("  BIAS REDUCTION SUMMARY")
        print("=" * 60)
        for entry in self.reduction_log:
            print(f"\n  [{entry['strategy'].upper()}]")
            print(f"    Action: {entry['action']}")
            print(f"    Impact: {entry['impact']}")
        print("\n" + "=" * 60)
