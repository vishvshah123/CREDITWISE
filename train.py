import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import TARGET_COL, MODELS_DIR
from src.data_generator import generate_dataset
from src.preprocessor import Preprocessor
from src.feature_engineer import FeatureEngineer
from src.trainer import ModelTrainer

def run_training():
    print("Starting CreditWise v3 Pipeline (Realistic Synthetic Dataset)...")

    # 1. Generate realistic dataset from banking underwriting rules
    df = generate_dataset(n=3000)
    print(f"Dataset generated. Shape: {df.shape}, Approval rate: {df[TARGET_COL].mean():.1%}")

    # 2. Preprocess
    preprocessor = Preprocessor()
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.encode_categorical(df)
    print("Encoded categorical features.")

    # 3. Feature Engineering
    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)
    print("Engineered features: DTI, EMI-to-Income, Balance_Income.")

    df.fillna(0, inplace=True)

    # 4. Split & Scale
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = preprocessor.scale_features(X_train, fit=True)
    X_test  = preprocessor.scale_features(X_test,  fit=False)
    print(f"Split: train={X_train.shape}, test={X_test.shape}")

    # 5. Train models
    trainer = ModelTrainer()
    trainer.train_all(X_train, y_train)

    # 6. Evaluate & optimize threshold
    results = trainer.evaluate_all(X_test, y_test)
    print("\nModel Performance:")
    print(results.to_string(index=False))

    opt_thresh = trainer.optimize_threshold(X_test, y_test)
    print(f"\nBest Model: {trainer.best_model_name}")
    print(f"Optimal Threshold (F0.5): {opt_thresh:.4f}")

    # 7. Save artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    trainer.save_model()
    joblib.dump(preprocessor, f"{MODELS_DIR}/preprocessor.pkl")
    joblib.dump(engineer,     f"{MODELS_DIR}/feature_engineer.pkl")
    joblib.dump({
        'feature_cols': preprocessor.feature_cols,
        'optimal_threshold': float(opt_thresh)
    }, f"{MODELS_DIR}/pipeline_config.pkl")
    print("All artifacts saved.")

if __name__ == '__main__':
    run_training()
