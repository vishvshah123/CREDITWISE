"""
CreditWise - Threshold Optimizer: Find optimal decision threshold using PR curves.

WHY NOT 0.5? The default 0.5 threshold is arbitrary. In lending:
- False Positives (approve bad loan) cost ~$50K+ in losses
- False Negatives (reject good applicant) cost ~$5K in missed revenue
- So we MUST shift the threshold to minimize costly FPs.
"""
import numpy as np
from sklearn.metrics import (precision_recall_curve, f1_score,
                             fbeta_score, confusion_matrix)


class ThresholdOptimizer:
    def __init__(self, beta=0.5):
        """
        beta < 1 weights precision higher than recall.
        WHY beta=0.5? In lending, precision (not approving bad loans)
        is MORE important than recall (approving all good applicants).
        F0.5 gives 2x weight to precision over recall.
        """
        self.beta = beta
        self.optimal_threshold = 0.5
        self.pr_curve_data = None
        self.threshold_analysis = None
    
    def optimize(self, y_true, y_proba):
        """
        Find the threshold that maximizes F-beta score.
        
        HOW IT WORKS:
        1. Try every possible threshold from the PR curve
        2. At each threshold, compute F-beta score
        3. Pick the threshold with highest F-beta
        
        F-beta = (1 + beta²) × (precision × recall) / (beta² × precision + recall)
        With beta=0.5: precision counts 4× more than recall
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Store PR curve data for plotting
        self.pr_curve_data = {
            'precisions': precisions,
            'recalls': recalls,
            'thresholds': thresholds
        }
        
        # Compute F-beta at each threshold
        f_scores = []
        for p, r in zip(precisions[:-1], recalls[:-1]):
            if p + r > 0:
                fb = (1 + self.beta**2) * (p * r) / (self.beta**2 * p + r)
            else:
                fb = 0
            f_scores.append(fb)
        f_scores = np.array(f_scores)
        
        # Find optimal threshold
        best_idx = np.argmax(f_scores)
        self.optimal_threshold = thresholds[best_idx]
        
        # Analyze multiple thresholds for comparison
        analysis = []
        for t in [0.3, 0.4, 0.5, self.optimal_threshold, 0.6, 0.7]:
            y_pred = (y_proba >= t).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            fb = ((1 + self.beta**2) * prec * rec / 
                  (self.beta**2 * prec + rec)) if (self.beta**2 * prec + rec) > 0 else 0
            approval_rate = (tp + fp) / len(y_true)
            
            analysis.append({
                'threshold': round(t, 4),
                'precision': round(prec, 4),
                'recall': round(rec, 4),
                'f1': round(f1, 4),
                f'f{self.beta}': round(fb, 4),
                'approval_rate': round(approval_rate, 4),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn),
                'is_optimal': abs(t - self.optimal_threshold) < 0.001
            })
        
        self.threshold_analysis = analysis
        self._print_analysis()
        
        return self.optimal_threshold
    
    def _print_analysis(self):
        print("\n" + "=" * 60)
        print("  THRESHOLD OPTIMIZATION")
        print("=" * 60)
        print(f"\n  Optimization metric: F{self.beta} (precision-weighted)")
        print(f"  Optimal threshold: {self.optimal_threshold:.4f}")
        
        print(f"\n  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} "
              f"{'F'+str(self.beta):>10} {'Approval%':>10} {'FP':>6}")
        print("  " + "-" * 68)
        
        for row in self.threshold_analysis:
            marker = " ★" if row['is_optimal'] else ""
            print(f"  {row['threshold']:>10.4f} {row['precision']:>10.4f} "
                  f"{row['recall']:>10.4f} {row['f1']:>10.4f} "
                  f"{row[f'f{self.beta}']:>10.4f} {row['approval_rate']:>10.1%} "
                  f"{row['false_positives']:>6}{marker}")
        
        print(f"\n  BUSINESS IMPACT at optimal threshold ({self.optimal_threshold:.4f}):")
        opt = [r for r in self.threshold_analysis if r['is_optimal']][0]
        default = [r for r in self.threshold_analysis if r['threshold'] == 0.5]
        if default:
            d = default[0]
            fp_reduction = d['false_positives'] - opt['false_positives']
            fn_increase = opt['false_negatives'] - d['false_negatives']
            print(f"    vs. default (0.5):")
            print(f"    • {fp_reduction} fewer risky approvals (FP reduction)")
            print(f"    • {fn_increase} more good applicants rejected (FN increase)")
            print(f"    • Net: Better risk management at cost of some lost customers")
        
        print("=" * 60)
    
    def apply_threshold(self, y_proba):
        """Apply the optimal threshold to get predictions."""
        return (y_proba >= self.optimal_threshold).astype(int)
