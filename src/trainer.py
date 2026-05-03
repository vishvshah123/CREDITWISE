import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
import joblib
from src.config import MODELS_DIR

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
            # Restrict depth and features to force trees to learn from other financial parameters, not just Credit_History
            'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=6, max_features=0.4, min_samples_leaf=4, class_weight='balanced', random_state=42),
            'XGBoost': XGBClassifier(eval_metric='logloss', max_depth=4, colsample_bytree=0.5, subsample=0.8, random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        self.optimal_threshold = 0.5
        
    def train_all(self, X_train, y_train, sample_weights=None):
        for name, model in self.models.items():
            if sample_weights is not None:
                model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train, y_train)

    def evaluate_all(self, X_test, y_test):
        results = []
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            res = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_proba)
            }
            results.append(res)
            
        df_results = pd.DataFrame(results)
        
        # Select best based on F1 score
        best_idx = df_results['F1'].idxmax()
        self.best_model_name = df_results.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]
        
        return df_results

    def optimize_threshold(self, X_test, y_test, beta=0.5):
        """
        Find optimal threshold favoring Precision (beta=0.5).
        Approving bad loans (False Positive) is worse than rejecting good ones (False Negative).
        """
        y_proba = self.best_model.predict_proba(X_test)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
        
        fscores = []
        for p, r in zip(precisions, recalls):
            if p + r == 0:
                fscores.append(0)
            else:
                # F-beta score
                fscores.append((1 + beta**2) * (p * r) / ((beta**2 * p) + r))
                
        # Drop last F-score because thresholds length is len(precisions)-1
        fscores = fscores[:-1]
        best_idx = np.argmax(fscores)
        self.optimal_threshold = thresholds[best_idx]
        return self.optimal_threshold

    def save_model(self, filepath=None):
        if filepath is None:
            filepath = f"{MODELS_DIR}/best_model.pkl"
        joblib.dump(self.best_model, filepath)
