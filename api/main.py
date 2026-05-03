from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.config import MODELS_DIR
from src.explainer import Explainer
from src.business_rules import apply_rules

app = FastAPI(title="CreditWise Loan Approval API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def preprocess_input(app_data: LoanApplication):
    """Common preprocessing pipeline. Returns scaled DataFrame."""
    df = pd.DataFrame([{
        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
        for k, v in app_data.model_dump().items()
    }])
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.encode_categorical(df)
    df = engineer.engineer_features(df)
    df.fillna(0, inplace=True)
    if 'Gender' in df.columns:
        df.drop('Gender', axis=1, inplace=True)
    for col in config['feature_cols']:
        if col not in df.columns:
            df[col] = 0
    df = df[config['feature_cols']]
    X_scaled = preprocessor.scale_features(df, fit=False)
    return X_scaled

@app.get("/")
def read_root():
    return {"message": "CreditWise API v2.0 — POST /predict or /counterfactual"}

@app.post("/predict")
def predict_loan(app_data: LoanApplication):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        X_scaled = preprocess_input(app_data)
        ml_prob = float(model.predict_proba(X_scaled)[0, 1])

        # Apply hard business rules on top of ML score
        rules_result = apply_rules(app_data.model_dump(), ml_prob)
        final_prob = rules_result['adjusted_probability']
        approved = bool(final_prob >= opt_threshold)

        explainer = Explainer(model, X_scaled, config['feature_cols'])
        contributions = explainer.get_local_explanation(X_scaled)

        return {
            "prediction": "Approved" if approved else "Rejected",
            "probability": round(final_prob, 4),
            "ml_probability": round(ml_prob, 4),
            "risk_score": round(1 - final_prob, 4),
            "threshold_used": float(opt_threshold),
            "rules_triggered": rules_result['rules_triggered'],
            "underwriting": {
                "dti": rules_result['dti'],
                "income_coverage": rules_result['income_coverage'],
                "emi": rules_result['emi'],
            },
            "explanation": {
                "top_factors": contributions[:5],
                "all_factors": contributions
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/counterfactual")
def get_counterfactual(app_data: LoanApplication):
    """Returns actionable recommendations: what to change to flip a rejection to approval."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        X_scaled = preprocess_input(app_data)
        prob = model.predict_proba(X_scaled)[0, 1]
        approved = bool(prob >= opt_threshold)

        # Get all SHAP contributions
        explainer = Explainer(model, X_scaled, config['feature_cols'])
        contributions = explainer.get_local_explanation(X_scaled)

        suggestions = []
        raw = app_data.model_dump()

        if not approved:
            # Find top negative factors and give specific advice
            negative = [c for c in contributions if c['impact'] < 0]
            for factor in negative[:3]:
                feat = factor['feature']
                advice = _generate_advice(feat, raw)
                if advice:
                    suggestions.append(advice)

        return {
            "currently_approved": approved,
            "current_probability": round(float(prob), 4),
            "gap_to_approval": round(float(opt_threshold - prob), 4) if not approved else 0,
            "recommendations": suggestions
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def _generate_advice(feature: str, raw: dict) -> dict | None:
    """Map a feature name to a human-readable improvement suggestion."""
    advice_map = {
        "Credit_Utilization": {
            "field": "Credit_Utilization",
            "current": raw.get("Credit_Utilization"),
            "target": max(0, raw.get("Credit_Utilization", 80) - 30),
            "action": "Reduce credit utilization",
            "detail": "Pay down revolving balances to get utilization below 30%."
        },
        "EMI_to_Income": {
            "field": "ApplicantIncome",
            "current": raw.get("ApplicantIncome"),
            "target": round(raw.get("ApplicantIncome", 3000) * 1.25),
            "action": "Increase monthly income",
            "detail": "Your EMI-to-Income ratio is high. Increasing income or adding a co-applicant improves this ratio."
        },
        "DTI": {
            "field": "Existing_Debt",
            "current": raw.get("Existing_Debt"),
            "target": round(raw.get("Existing_Debt", 5000) * 0.5),
            "action": "Reduce existing debt",
            "detail": "Pay off existing loans to lower your Debt-to-Income ratio below 40%."
        },
        "Balance_Income": {
            "field": "CoapplicantIncome",
            "current": raw.get("CoapplicantIncome"),
            "target": round(raw.get("ApplicantIncome", 3000) * 0.5),
            "action": "Add a co-applicant",
            "detail": "A co-applicant with income improves your balance income significantly."
        },
        "Employment_Stability": {
            "field": "Employment_Stability",
            "current": raw.get("Employment_Stability"),
            "target": max(raw.get("Employment_Stability", 1) + 2, 3),
            "action": "Build employment tenure",
            "detail": "Lenders prefer 3+ years at current employer as a stability signal."
        },
        "LoanAmount_Log": {
            "field": "LoanAmount",
            "current": raw.get("LoanAmount"),
            "target": round(raw.get("LoanAmount", 200) * 0.8),
            "action": "Reduce loan amount",
            "detail": "Requesting a smaller loan amount reduces your risk profile."
        },
        "EMI": {
            "field": "LoanAmount",
            "current": raw.get("LoanAmount"),
            "target": round(raw.get("LoanAmount", 200) * 0.8),
            "action": "Reduce loan amount or extend term",
            "detail": "A lower EMI improves your monthly budget ratio significantly."
        },
    }
    return advice_map.get(feature)
