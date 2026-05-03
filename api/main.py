from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import sys

# Add project root to sys path to allow importing src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.config import MODELS_DIR
from src.explainer import Explainer

app = FastAPI(title="CreditWise Loan Approval API", version="1.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts at startup
try:
    model = joblib.load(f"{MODELS_DIR}/best_model.pkl")
    preprocessor = joblib.load(f"{MODELS_DIR}/preprocessor.pkl")
    engineer = joblib.load(f"{MODELS_DIR}/feature_engineer.pkl")
    config = joblib.load(f"{MODELS_DIR}/pipeline_config.pkl")
    opt_threshold = config['optimal_threshold']
except Exception as e:
    print(f"Error loading models: {e}. Please run train.py first.")
    model, preprocessor, engineer, config, opt_threshold = None, None, None, None, 0.5

class LoanApplication(BaseModel):
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str
    Existing_Debt: float
    Credit_Utilization: float
    Employment_Stability: float
    Loan_Purpose: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the CreditWise API. Use POST /predict to score applications."}

@app.post("/predict")
def predict_loan(app_data: LoanApplication):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
        
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([app_data.model_dump()])
        
        # Keep original copy for Explainer
        X_raw = df.copy()
        
        # Preprocess
        df = preprocessor.handle_missing_values(df)
        df = preprocessor.encode_categorical(df)
        df = engineer.engineer_features(df)
        df.fillna(0, inplace=True)
        
        if 'Gender' in df.columns:
            df.drop('Gender', axis=1, inplace=True)
            
        # Ensure exact column order and presence
        for col in config['feature_cols']:
            if col not in df.columns:
                df[col] = 0
        df = df[config['feature_cols']]
            
        X_scaled = preprocessor.scale_features(df, fit=False)
        
        # Predict
        prob = model.predict_proba(X_scaled)[0, 1]
        approved = bool(prob >= opt_threshold)
        
        # Provide fallback simple explanation for now
        # Explaining single instances efficiently
        explainer = Explainer(model, X_scaled, config['feature_cols'])
        contributions = explainer.get_local_explanation(X_scaled)
        
        top_factors = contributions[:3]
        
        return {
            "prediction": "Approved" if approved else "Rejected",
            "probability": float(prob),
            "risk_score": float(1 - prob),
            "threshold_used": float(opt_threshold),
            "explanation": {
                "top_factors": top_factors
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
