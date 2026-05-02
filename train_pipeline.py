"""
CreditWise - End-to-End Training Pipeline
==========================================
Pipeline Flow: Dataset → Bias Detection → Neutralize → Train → Optimize → Explain

This script orchestrates the complete pipeline with reasoning at every step.
"""
import sys
import os
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.generate_data import generate_loan_dataset, save_dataset
from pipeline.data_loader import DataLoader
from pipeline.preprocessor import Preprocessor
from pipeline.feature_engineer import FeatureEngineer
from pipeline.bias_detector import BiasDetector
from pipeline.bias_reducer import BiasReducer
from pipeline.model_trainer import ModelTrainer
from pipeline.threshold_optimizer import ThresholdOptimizer
from pipeline.explainer import Explainer
from pipeline.recommender import Recommender


def run_pipeline():
    """Execute the full CreditWise training pipeline."""
    
    print("\n" + "=" * 60)
    print("  CREDITWISE LOAN APPROVAL SYSTEM")
    print("  End-to-End ML Pipeline")
    print("=" * 60)
    
    # ================================================================
    # STEP 1: GENERATE DATASET
    # REASONING: We use synthetic data with known biases so we can
    # validate that our bias detection actually finds them.
    # ================================================================
    print("\n\n" + "=" * 60)
    print("  STEP 1: DATASET GENERATION")
    print("=" * 60)
    
    df = generate_loan_dataset(n_samples=5000, bias_strength=0.15)
    save_dataset(df, output_dir='data')
    
    # ================================================================
    # STEP 2: LOAD & INSPECT DATA
    # ================================================================
    print("\n\n" + "=" * 60)
    print("  STEP 2: DATA LOADING & INSPECTION")
    print("=" * 60)
    
    loader = DataLoader()
    df = loader.load_from_dataframe(df)
    loader.print_report()
    
    # ================================================================
    # STEP 3: BIAS DETECTION (BEFORE any changes)
    # REASONING: We must MEASURE bias before we can fix it.
    # This establishes a baseline to compare against after mitigation.
    # ================================================================
    print("\n\n" + "=" * 60)
    print("  STEP 3: BIAS DETECTION (PRE-MITIGATION)")
    print("=" * 60)
    
    bias_detector = BiasDetector(protected_attributes=['gender'])
    pre_bias = bias_detector.detect_data_bias(df)
    
    # ================================================================
    # STEP 4: HANDLE MISSING VALUES FIRST
    # REASONING: Feature engineering involves divisions (DTI, LTV etc).
    # If income/expenses/property are NaN, the ratios become NaN too.
    # So we MUST impute BEFORE engineering features.
    # ================================================================
    print("\n\n" + "=" * 60)
    print("  STEP 4: MISSING VALUE TREATMENT")
    print("=" * 60)
    
    preprocessor = Preprocessor()
    df = preprocessor.handle_missing_values(df)
    
    # ================================================================
    # STEP 5: FEATURE ENGINEERING
    # REASONING: Create domain features AFTER imputation so all
    # ratios/divisions produce valid numbers, not NaN.
    # ================================================================
    print("\n\n" + "=" * 60)
    print("  STEP 5: FEATURE ENGINEERING")
    print("=" * 60)
    
    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)
    engineer.print_reasoning()
    
    # Safety: fill any remaining NaN from edge cases in engineering
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        print(f"  Filling {nan_count} remaining NaN values with 0 (edge cases)")
        df = df.fillna(0)
    
    # ================================================================
    # STEP 6: BIAS REDUCTION (NEUTRALIZE)
    # REASONING: We reduce bias at the DATA level before training:
    # 1. Identify proxy features correlated with gender
    # 2. Compute sample weights to equalize outcomes across groups
    # ================================================================
    print("\n\n" + "=" * 60)
    print("  STEP 6: BIAS REDUCTION (NEUTRALIZATION)")
    print("=" * 60)
    
    reducer = BiasReducer()
    
    # Step 6a: Detect proxy features
    df, proxies = reducer.remove_proxy_features(df, 'gender', threshold=0.25)
    
    # Step 6b: Compute reweighting
    sample_weights_full = reducer.compute_reweighting(df, 'gender', 'loan_approved')
    
    # Step 6c: Get class weights
    class_weights = reducer.get_balanced_class_weights(df['loan_approved'])
    
    reducer.print_summary()
    
    # ================================================================
    # STEP 7: ENCODING, SCALING, SPLIT
    # ================================================================
    print("\n\n" + "=" * 60)
    print("  STEP 7: ENCODING, SCALING & SPLIT")
    print("=" * 60)
    
    # Encode categorical features
    print("\n  Encoding categorical features...")
    df = preprocessor.encode_categorical(df)
    
    # Remove protected attributes
    protected_to_remove = ['gender']
    for col in protected_to_remove:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            print(f"  Removed protected attribute: {col}")
    
    # Detect and handle outliers
    print("  Detecting outliers...")
    df, outlier_report = preprocessor.detect_outliers(df)
    
    # Split
    print("  Train/test split (80/20 stratified)...")
    target_col = 'loan_approved'
    
    # Store gender/age before splitting
    gender_col = None
    age_col = df['age'].copy() if 'age' in df.columns else None
    # gender was already removed; get from original data
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    preprocessor.feature_cols = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    print("  Scaling numeric features...")
    X_train = preprocessor.scale_features(X_train, target_col=None, fit=True)
    X_test = preprocessor.scale_features(X_test, target_col=None, fit=False)
    
    # Final NaN safety check
    if X_train.isnull().sum().sum() > 0:
        X_train = X_train.fillna(0)
    if X_test.isnull().sum().sum() > 0:
        X_test = X_test.fillna(0)
    
    preprocessor.print_reasoning()
    
    # Build metadata for bias analysis
    metadata = {}
    # Reload gender from original dataset for bias testing
    original_df = generate_loan_dataset(n_samples=5000, bias_strength=0.15)
    metadata['gender_train'] = original_df['gender'].iloc[X_train.index]
    metadata['gender_test'] = original_df['gender'].iloc[X_test.index]
    
    # Align sample weights with training indices
    train_indices = X_train.index
    sample_weights = sample_weights_full[train_indices]
    
    print(f"\n  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    print(f"  Features: {X_train.columns.tolist()}")
    
    # ================================================================
    # STEP 8: MODEL TRAINING (with bias-reduction weights)
    # REASONING: We pass sample_weights from Step 6 to the models.
    # This makes the models see an "unbiased" version of the data.
    # ================================================================
    print("\n\n" + "=" * 60)
    print("  STEP 8: MODEL TRAINING (BIAS-AWARE)")
    print("=" * 60)
    
    trainer = ModelTrainer()
    trainer.train_all(X_train, y_train, sample_weights=sample_weights)
    results = trainer.evaluate_all(X_test, y_test)
    
    # ================================================================
    # STEP 9: THRESHOLD OPTIMIZATION
    # REASONING: Find threshold that maximizes F0.5 (precision-weighted)
    # to minimize approving risky loans.
    # ================================================================
    print("\n\n" + "=" * 60)
    print("  STEP 9: THRESHOLD OPTIMIZATION")
    print("=" * 60)
    
    optimizer = ThresholdOptimizer(beta=0.5)
    y_proba = trainer.best_model.predict_proba(X_test)[:, 1]
    optimal_threshold = optimizer.optimize(y_test.values, y_proba)
    
    # Apply optimized threshold
    y_pred_optimized = optimizer.apply_threshold(y_proba)
    
    # ================================================================
    # STEP 9: BIAS DETECTION (POST-MITIGATION)
    # REASONING: Check if our bias reduction actually worked.
    # Compare pre vs post metrics.
    # ================================================================
    print("\n\n" + "=" * 60)
    print("  STEP 10: BIAS DETECTION (POST-MITIGATION)")
    print("=" * 60)
    
    protected_test = {}
    if 'gender_test' in metadata:
        protected_test['gender_test'] = metadata['gender_test']
    
    post_bias = bias_detector.detect_model_bias(
        y_test.values, y_pred_optimized, protected_test
    )
    
    # ================================================================
    # STEP 10: EXPLAINABILITY (SHAP + LIME)
    # ================================================================
    print("\n\n" + "=" * 60)
    print("  STEP 11: EXPLAINABILITY (SHAP + LIME)")
    print("=" * 60)
    
    explainer = Explainer(
        trainer.best_model, X_train, X_train.columns.tolist()
    )
    
    # Global importance
    print("\n  Computing global feature importance...")
    global_importance, shap_values_all = explainer.get_global_importance(
        X_test.iloc[:200]
    )
    print("\n  Global Feature Importance (SHAP):")
    for _, row in global_importance.head(10).iterrows():
        bar = "#" * int(row['mean_abs_shap'] * 100)
        print(f"    {row['feature']:30s} {row['mean_abs_shap']:.4f} {bar}")
    
    # ================================================================
    # STEP 11: SAVE EVERYTHING
    # ================================================================
    print("\n\n" + "=" * 60)
    print("  STEP 12: SAVING ARTIFACTS")
    print("=" * 60)
    
    os.makedirs('models', exist_ok=True)
    
    # Save model
    trainer.save_best_model('models/best_model.pkl')
    
    # Save pipeline components
    pipeline_data = {
        'preprocessor': preprocessor,
        'feature_engineer': engineer,
        'bias_detector': bias_detector,
        'bias_reducer': reducer,
        'threshold_optimizer': optimizer,
        'feature_names': X_train.columns.tolist(),
        'metadata': metadata,
        'pre_bias_report': pre_bias,
        'post_bias_report': post_bias,
        'model_results': results,
        'global_importance': global_importance,
        'optimal_threshold': optimal_threshold,
        'class_weights': class_weights
    }
    joblib.dump(pipeline_data, 'models/pipeline_data.pkl')
    print("  💾 Pipeline data saved to: models/pipeline_data.pkl")
    
    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n\n" + "=" * 60)
    print("  PIPELINE COMPLETE - SUMMARY")
    print("=" * 60)
    
    print(f"\n  Best Model: {trainer.best_model_name}")
    print(f"  Optimal Threshold: {optimal_threshold:.4f}")
    print(f"  AUC-ROC: {results.loc[results['Model'] == trainer.best_model_name, 'AUC-ROC'].values[0]:.4f}")
    
    if pre_bias and 'gender' in pre_bias:
        pre_dpd = pre_bias['gender']['demographic_parity_diff']
        print(f"\n  Bias Reduction Results:")
        print(f"    Pre-mitigation DPD: {pre_dpd:.4f}")
        if post_bias and 'gender' in post_bias:
            rates = [m['approval_rate'] for m in post_bias['gender']['group_metrics'].values()]
            post_dpd = max(rates) - min(rates) if rates else 0
            print(f"    Post-mitigation DPD: {post_dpd:.4f}")
            improvement = (1 - post_dpd / pre_dpd) * 100 if pre_dpd > 0 else 0
            print(f"    Improvement: {improvement:.1f}%")
    
    print(f"\n  Run 'streamlit run app.py' to launch the dashboard!")
    print("=" * 60)


if __name__ == '__main__':
    run_pipeline()
