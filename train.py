import os
import joblib
from sklearn.model_selection import train_test_split
from src.config import TARGET_COL, MODELS_DIR
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.feature_engineer import FeatureEngineer
from src.bias_mitigator import BiasMitigator
from src.trainer import ModelTrainer

def run_training():
    print("Starting CreditWise Pipeline...")
    
    # 1. Load Data
    loader = DataLoader()
    df = loader.load_data()
    print("Data Loaded. Shape:", df.shape)
    
    # 2. Preprocess
    preprocessor = Preprocessor()
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.encode_categorical(df)
    print("Missing values imputed and categorical features encoded.")
    
    # 3. Feature Engineering
    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)
    print("Engineered features (Total_Income, EMI, Balance_Income).")
    
    # Fill remaining NaNs (if any division by zero)
    df.fillna(0, inplace=True)
    
    # User requested removing gender/bias logic entirely to trust the features
    # 4. Train/Test Split & Scaling
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train = preprocessor.scale_features(X_train, fit=True)
    X_test = preprocessor.scale_features(X_test, fit=False)
    print(f"Data split and scaled. Training size: {X_train.shape}")
    
    # 5. Model Training
    trainer = ModelTrainer()
    trainer.train_all(X_train, y_train)
    
    # 6. Evaluation & Optimization
    results = trainer.evaluate_all(X_test, y_test)
    print("\nModel Performance:")
    print(results.to_string(index=False))
    
    opt_thresh = trainer.optimize_threshold(X_test, y_test)
    print(f"\nBest Model: {trainer.best_model_name}")
    print(f"Optimal Threshold (F0.5): {opt_thresh:.4f}")
    
    # Save Artifacts
    trainer.save_model()
    joblib.dump(preprocessor, f"{MODELS_DIR}/preprocessor.pkl")
    joblib.dump(engineer, f"{MODELS_DIR}/feature_engineer.pkl")
    joblib.dump({'feature_cols': preprocessor.feature_cols, 'optimal_threshold': opt_thresh}, f"{MODELS_DIR}/pipeline_config.pkl")
    print("All artifacts saved to models/ directory.")

if __name__ == '__main__':
    run_training()
