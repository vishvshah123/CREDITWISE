# CreditWise – AI Decision Intelligence Platform

A production-ready end-to-end Machine Learning and Underwriting Intelligence pipeline that predicts loan approvals based on applicant financial and demographic data. Built with an emphasis on **Explainable AI (SHAP)**, **Hard Banking Underwriting Guardrails**, **Real-Time What-If Simulation**, and a clean decoupled architecture (FastAPI Backend + Modern Responsive Vanilla HTML/CSS/JS Frontend).

---

## Live Deployments

- **Live Application (Frontend):** [https://vishvshah123.github.io/CREDITWISE/](https://vishvshah123.github.io/CREDITWISE/)
- **Live Backend API (Render):** [https://creditwise-api.onrender.com](https://creditwise-api.onrender.com)
- **Interactive Swagger Documentation:** [https://creditwise-api.onrender.com/docs](https://creditwise-api.onrender.com/docs)
- **Health Check Endpoint:** [https://creditwise-api.onrender.com/health](https://creditwise-api.onrender.com/health)

> **Note on Free-Tier Hosting:** Render free instances spin down after inactivity. The first request may take ~30–45 seconds to wake up; subsequent requests respond instantaneously.

---

## Key Features

- **Explainable AI (SHAP Waterfall Analysis):** Real-time local feature contribution breakdown for every applicant prediction, showing exactly how each financial factor influenced the approval score.
- **Dual-Layer Decision Architecture:** Combines trained Machine Learning classification probabilities with hard underwriting rules (Debt-to-Income caps, credit utilization ceilings, and minimum income coverage).
- **Counterfactual Action Engine:** When an application is rejected, dynamically computes actionable next steps (e.g. paying down balances or adding a co-applicant) required to bridge the gap to approval.
- **Interactive What-If Simulator:** Real-time parameter sliders allowing loan officers and applicants to simulate changes in income, loan term, debt, or utilization with live recalculation.
- **Fully Responsive Modern UI:** Premium glassmorphism dark interface optimized for desktop, tablet, and mobile browsers.

---

## System Architecture

```
CREDITWISE/
├── api/
│   └── main.py              # FastAPI application (predict, counterfactual, health, docs)
├── data/
│   ├── loan_applications.csv
│   └── real_loan_dataset.csv
├── frontend/
│   ├── index.html           # Semantic HTML5 application structure
│   ├── styles.css           # Premium dark theme & responsive media queries
│   └── app.js               # Client logic, Chart.js SHAP charts, debounce What-If
├── models/
│   ├── best_model.pkl       # Serialized Random Forest classifier
│   ├── feature_engineer.pkl # Serialized feature transformation pipeline
│   ├── pipeline_config.pkl  # Optimal threshold & feature column mappings
│   ├── pipeline_data.pkl    # Reference distributions
│   └── preprocessor.pkl     # Standard scalers and label encoders
├── src/
│   ├── bias_mitigator.py    # Demographic parity & reweighting
│   ├── business_rules.py    # Hard underwriting caps (DTI, utilization, coverage)
│   ├── config.py            # Path resolution and column schemas
│   ├── data_generator.py    # Synthetic banking data generator
│   ├── data_loader.py       # Dataset ingestion utilities
│   ├── explainer.py         # SHAP TreeExplainer & LIME tabular integrations
│   ├── feature_engineer.py  # Financial feature engineering (DTI, EMI, Balance Income)
│   ├── preprocessor.py      # Missing value imputation & standardization
│   └── trainer.py           # Model training, evaluation & threshold optimization
├── render.yaml              # Render Blueprint for automated API & frontend provisioning
├── requirements.txt         # Pinned backend dependencies
└── train.py                 # Pipeline retraining script
```

---

## Local Development & Setup

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/vishvshah123/CREDITWISE.git
cd CREDITWISE
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Backend API
```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```
- API Root: `http://127.0.0.1:8000`
- Swagger Docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

### 3. Launch Frontend
The frontend automatically detects local environments and connects to `http://127.0.0.1:8000`:
```bash
python -m http.server 8080 --directory frontend
```
Open `http://localhost:8080` in any browser.

---

## API Reference

### `POST /predict`
Evaluates applicant profile, returns approval determination, calibrated probability, underwriting metrics, rules triggered, and SHAP feature impacts.

### `POST /counterfactual`
Computes specific prescriptive targets to flip a rejection into an approval.

### `GET /health`
Returns service health, model load status, and active decision threshold.

---

## Deployment Configuration

- **Backend (Render Web Service):** Managed via `render.yaml` with Python 3.10+, executing `uvicorn api.main:app --host 0.0.0.0 --port $PORT`.
- **Frontend (GitHub Pages / Render Static Site):** Serves `frontend/` statically with automatic CORS negotiation.
