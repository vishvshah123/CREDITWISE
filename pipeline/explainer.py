"""
CreditWise - Explainer: SHAP + LIME explanations for every prediction.

WHY BOTH SHAP AND LIME?
- SHAP: Mathematically grounded (Shapley values from game theory), globally consistent
- LIME: Simpler, faster, creates local linear approximations
- Together: Cross-validate explanations. If both agree, high confidence.
- Regulatory compliance: "Right to explanation" laws (GDPR, ECOA) require this.
"""
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings('ignore')


class Explainer:
    def __init__(self, model, X_train, feature_names):
        """
        Initialize SHAP and LIME explainers.
        
        WHY initialize on X_train?
        - SHAP needs a background dataset to compute expected values
        - LIME needs training distribution to generate perturbations
        - Using X_train (not X_test) avoids data leakage in explanations
        """
        self.model = model
        self.feature_names = feature_names
        self.X_train = X_train
        
        # SHAP explainer
        # Use TreeExplainer for tree models (fast), KernelExplainer for others
        model_type = type(model).__name__
        if model_type in ['XGBClassifier', 'RandomForestClassifier', 
                          'GradientBoostingClassifier']:
            self.shap_explainer = shap.TreeExplainer(model)
            self.shap_type = 'Tree'
        else:
            # KernelExplainer for model-agnostic explanations
            background = shap.sample(X_train, min(100, len(X_train)))
            self.shap_explainer = shap.KernelExplainer(model.predict_proba, background)
            self.shap_type = 'Kernel'
        
        # LIME explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=feature_names,
            class_names=['Rejected', 'Approved'],
            mode='classification',
            discretize_continuous=True  # WHY: Makes explanations more interpretable
        )
        
        print(f"  Explainer initialized:")
        print(f"    SHAP: {self.shap_type}Explainer")
        print(f"    LIME: LimeTabularExplainer")
        print(f"    Features: {len(feature_names)}")
    
    def explain_shap(self, X, max_samples=None):
        """
        Get SHAP values for predictions.
        
        WHAT SHAP VALUES MEAN:
        - Positive SHAP -> pushes prediction toward APPROVAL
        - Negative SHAP -> pushes prediction toward REJECTION
        - Magnitude = how much that feature matters for THIS specific person
        """
        if max_samples and len(X) > max_samples:
            X = X.iloc[:max_samples] if hasattr(X, 'iloc') else X[:max_samples]
        
        shap_output = self.shap_explainer(X)
        
        # Handle SHAP Explanation objects (newer SHAP versions)
        if hasattr(shap_output, 'values'):
            shap_values = shap_output.values
        else:
            shap_values = shap_output
        
        # For binary classification, SHAP may return 3D array (samples, features, classes)
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]  # Take positive class
        elif isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        return shap_values
    
    def explain_lime_single(self, instance, num_features=10):
        """
        Get LIME explanation for a single prediction.
        
        HOW LIME WORKS:
        1. Take the instance to explain
        2. Generate ~5000 perturbed (slightly modified) versions
        3. Predict each perturbed version with the real model
        4. Fit a simple linear model on the perturbations
        5. The linear coefficients = feature importances
        
        WHY LIME IS USEFUL:
        - Simpler to understand than SHAP for stakeholders
        - Shows "if this feature changed, prediction would change by X"
        - Fast for individual explanations
        """
        if hasattr(instance, 'values'):
            instance = instance.values.flatten()
        
        explanation = self.lime_explainer.explain_instance(
            instance, 
            self.model.predict_proba,
            num_features=num_features,
            num_samples=5000
        )
        
        return explanation
    
    def get_feature_contributions(self, instance, shap_values_single):
        """
        Get a human-readable breakdown of feature contributions.
        
        Returns a sorted list of (feature, value, shap_value, direction)
        showing which features pushed toward approval vs rejection.
        """
        contributions = []
        
        if hasattr(instance, 'values'):
            values = instance.values.flatten()
        else:
            values = instance.flatten()
        
        for i, (feat, val, sv) in enumerate(zip(
            self.feature_names, values, shap_values_single
        )):
            contributions.append({
                'feature': feat,
                'value': float(val),
                'shap_value': float(sv),
                'direction': 'Toward Approval' if sv > 0 else 'Toward Rejection',
                'abs_impact': abs(float(sv))
            })
        
        # Sort by absolute impact (most impactful first)
        contributions.sort(key=lambda x: x['abs_impact'], reverse=True)
        return contributions
    
    def get_global_importance(self, X_sample=None):
        """
        Get global feature importance from SHAP (across all predictions).
        
        WHY GLOBAL IMPORTANCE?
        While individual explanations tell us about ONE person,
        global importance tells us what features matter OVERALL.
        This helps identify if the model relies on potentially biased features.
        """
        if X_sample is None:
            X_sample = self.X_train.iloc[:min(500, len(self.X_train))]
        
        shap_values = self.explain_shap(X_sample)
        
        # Mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        return importance, shap_values
    
    def compare_shap_lime(self, instance):
        """
        Compare SHAP and LIME explanations for the same instance.
        
        WHY COMPARE?
        - If both methods agree on top features → high confidence
        - If they disagree → the explanation may be unstable
        - Builds trust with stakeholders by showing consistency
        """
        # SHAP
        shap_values = self.explain_shap(
            pd.DataFrame([instance.values], columns=self.feature_names)
            if hasattr(instance, 'values')
            else pd.DataFrame([instance], columns=self.feature_names)
        )
        shap_contributions = self.get_feature_contributions(instance, shap_values[0])
        
        # LIME
        lime_exp = self.explain_lime_single(instance)
        lime_features = {feat: weight for feat, weight in lime_exp.as_list()}
        
        # Top 5 from each
        top_shap = [c['feature'] for c in shap_contributions[:5]]
        
        return {
            'shap_top5': top_shap,
            'shap_contributions': shap_contributions[:10],
            'lime_explanation': lime_exp,
            'lime_features': lime_features
        }
