import shap
import lime
import lime.lime_tabular
import numpy as np

class Explainer:
    def __init__(self, model, X_train, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.X_train = X_train
        
        # Setup SHAP
        model_type = type(model).__name__
        if model_type in ['XGBClassifier', 'RandomForestClassifier']:
            self.shap_explainer = shap.TreeExplainer(model)
            self.is_tree = True
        else:
            background = shap.sample(X_train, min(100, len(X_train)))
            self.shap_explainer = shap.KernelExplainer(model.predict_proba, background)
            self.is_tree = False
            
        # Setup LIME
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=feature_names,
            class_names=['Rejected', 'Approved'],
            mode='classification'
        )

    def explain_shap(self, X):
        shap_output = self.shap_explainer(X)
        if hasattr(shap_output, 'values'):
            shap_vals = shap_output.values
        else:
            shap_vals = shap_output
            
        if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            return shap_vals[:, :, 1]
        elif isinstance(shap_vals, list):
            return shap_vals[1]
        return shap_vals
        
    def get_local_explanation(self, instance):
        """Returns feature contributions for a single instance."""
        sv = self.explain_shap(instance)
        if len(sv.shape) == 2:
            sv = sv[0]
            
        contributions = []
        vals = instance.iloc[0].values
        for feat, val, s_val in zip(self.feature_names, vals, sv):
            contributions.append({
                'feature': feat,
                'value': float(val),
                'impact': float(s_val)
            })
            
        # Sort by absolute impact
        contributions.sort(key=lambda x: abs(x['impact']), reverse=True)
        return contributions
