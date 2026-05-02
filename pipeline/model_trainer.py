"""
CreditWise - Model Trainer: Train and compare multiple classification models.
Each model choice is documented with reasoning for WHY it's included.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import xgboost as xgb
import joblib
import os


class ModelTrainer:
    """
    Train multiple models and compare performance.
    
    WHY MULTIPLE MODELS?
    - Different models have different strengths/weaknesses
    - Logistic Regression: interpretable, baseline
    - Random Forest: handles non-linearity, robust
    - XGBoost: best performance on tabular data
    - Gradient Boosting: alternative ensemble perspective
    We pick the best one based on our specific metric priority (precision).
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None
    
    def _get_models(self, class_weights=None):
        """
        Define models with reasoning for each hyperparameter choice.
        
        REASONING for key hyperparameters:
        - class_weight='balanced': Handles imbalanced approval/rejection ratio
        - max_depth=6: Prevents overfitting on training data
        - n_estimators=200: Enough trees for stable predictions
        - learning_rate=0.1: Standard starting point for boosting
        """
        models = {
            'Logistic Regression': {
                'model': LogisticRegression(
                    class_weight='balanced',  # WHY: Compensate for class imbalance
                    max_iter=1000,             # WHY: Ensure convergence
                    random_state=42,
                    C=1.0                      # WHY: Default regularization strength
                ),
                'reasoning': (
                    'Baseline model. Inherently interpretable — regulators prefer it. '
                    'Coefficients directly show feature importance direction and magnitude. '
                    'class_weight=balanced handles the ~35/65 class imbalance.'
                )
            },
            'Random Forest': {
                'model': RandomForestClassifier(
                    n_estimators=200,       # WHY: Enough trees for stability
                    max_depth=8,            # WHY: Limit depth to prevent overfitting
                    min_samples_split=10,   # WHY: Require 10 samples to split
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1               # WHY: Use all CPU cores for speed
                ),
                'reasoning': (
                    'Handles non-linear feature interactions (e.g., income × loan_amount). '
                    'Robust to outliers and noise. Feature importance is built-in. '
                    'max_depth=8 prevents overfitting while capturing complex patterns.'
                )
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,            # WHY: XGBoost tends to overfit; keep shallow
                    learning_rate=0.1,      # WHY: Standard, balances speed vs accuracy
                    subsample=0.8,          # WHY: Row sampling prevents overfitting
                    colsample_bytree=0.8,   # WHY: Feature sampling adds diversity
                    scale_pos_weight=1,     # Will be computed from data
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False
                ),
                'reasoning': (
                    'State-of-the-art for tabular data. Wins most Kaggle competitions. '
                    'Built-in L1/L2 regularization prevents overfitting. '
                    'subsample=0.8 and colsample=0.8 add stochastic regularization.'
                )
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    min_samples_split=10,
                    random_state=42
                ),
                'reasoning': (
                    'Alternative to XGBoost with different optimization approach. '
                    'Often more robust with smaller datasets. '
                    'Provides a different bias-variance trade-off for comparison.'
                )
            }
        }
        return models
    
    def train_all(self, X_train, y_train, sample_weights=None, class_weights=None):
        """
        Train all models and store results.
        
        REASONING for sample_weights:
        These come from the BiasReducer — they upweight underrepresented
        group-outcome combinations to counteract historical bias in the data.
        """
        models_config = self._get_models(class_weights)
        
        print("\n" + "=" * 60)
        print("  MODEL TRAINING")
        print("=" * 60)
        
        for name, config in models_config.items():
            model = config['model']
            print(f"\n  Training: {name}")
            print(f"  Reasoning: {config['reasoning']}")
            
            # Set scale_pos_weight for XGBoost
            if name == 'XGBoost':
                neg = (y_train == 0).sum()
                pos = (y_train == 1).sum()
                model.set_params(scale_pos_weight=neg / pos if pos > 0 else 1)
            
            # Train with sample weights if provided (for bias reduction)
            if sample_weights is not None and hasattr(model, 'fit'):
                try:
                    model.fit(X_train, y_train, sample_weight=sample_weights)
                    print(f"  ✅ Trained with bias-reduction sample weights")
                except TypeError:
                    model.fit(X_train, y_train)
                    print(f"  ⚠️  Model doesn't support sample_weight, trained without")
            else:
                model.fit(X_train, y_train)
                print(f"  ✅ Trained (no sample weights)")
            
            self.models[name] = model
        
        print(f"\n  Total models trained: {len(self.models)}")
        print("=" * 60)
    
    def evaluate_all(self, X_test, y_test):
        """
        Evaluate all models and compare.
        
        REASONING for metric choices:
        - Accuracy: Overall correctness (can be misleading with imbalance)
        - Precision: "Of approved loans, how many won't default?" — KEY for banks
        - Recall: "Of all good applicants, how many did we approve?"
        - F1: Harmonic mean of precision & recall
        - AUC-ROC: Overall ranking quality (threshold-independent)
        
        FOR LENDING, PRECISION IS KING because approving a bad loan costs
        far more ($$$) than rejecting a good applicant.
        """
        print("\n" + "=" * 60)
        print("  MODEL EVALUATION RESULTS")
        print("=" * 60)
        
        results_list = []
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1': f1_score(y_test, y_pred, zero_division=0),
                'AUC-ROC': roc_auc_score(y_test, y_proba)
            }
            results_list.append(metrics)
            
            cm = confusion_matrix(y_test, y_pred)
            
            print(f"\n  {name}:")
            print(f"    Accuracy:  {metrics['Accuracy']:.4f}")
            print(f"    Precision: {metrics['Precision']:.4f}  ← KEY METRIC for lending")
            print(f"    Recall:    {metrics['Recall']:.4f}")
            print(f"    F1 Score:  {metrics['F1']:.4f}")
            print(f"    AUC-ROC:   {metrics['AUC-ROC']:.4f}")
            print(f"    Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
        
        self.results = pd.DataFrame(results_list)
        
        # Select best model by AUC-ROC (most robust metric)
        best_idx = self.results['AUC-ROC'].idxmax()
        self.best_model_name = self.results.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n  🏆 Best Model: {self.best_model_name} (AUC-ROC: {self.results.loc[best_idx, 'AUC-ROC']:.4f})")
        print("=" * 60)
        
        return self.results
    
    def save_best_model(self, filepath='models/best_model.pkl'):
        """Save the best model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.best_model,
            'model_name': self.best_model_name,
            'results': self.results
        }, filepath)
        print(f"\n  💾 Best model saved to: {filepath}")
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from the best model."""
        model = self.best_model
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            return None
        
        fi = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return fi
