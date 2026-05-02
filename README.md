# CreditWise Loan Approval System

A production-ready end-to-end Machine Learning pipeline that predicts loan approval based on applicant financial and demographic data. Built with an emphasis on Explainable AI (SHAP), Bias Mitigation, and a clean decoupled architecture (FastAPI Backend + Vanilla HTML/CSS Frontend).

## Architecture

- **`data/`**: Contains the raw Dream Housing Finance dataset.
- **`src/`**: Modular Python package handling Data Loading, Preprocessing, Feature Engineering, Bias Mitigation, Model Training, and Explanations.
- **`models/`**: Stores serialized artifacts (`best_model.pkl`, `preprocessor.pkl`, etc.).
- **`api/`**: FastAPI application that loads the models and serves a `/predict` endpoint.
- **`frontend/`**: Minimal, professional HTML/CSS/JS user interface.

## Prerequisites

- Python 3.9+
- The required dependencies in `requirements.txt`.

## Step-by-Step Run Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)
I have already pre-trained the model and saved it in the `models/` directory. If you wish to re-train the model, run:
```bash
python train.py
```
This will:
1. Load and clean the dataset.
2. Engineer financial features (Total Income, EMI, Balance Income).
3. Mitigate gender bias using sample reweighting.
4. Train Logistic Regression, Random Forest, and XGBoost models.
5. Select the best model and optimize its decision threshold using Precision-Recall tradeoffs to minimize high-risk approvals.
6. Save the artifacts to `models/`.

### 3. Start the Backend API
Start the FastAPI server using Uvicorn:
```bash
uvicorn api.main:app --reload
```
The API will be available at `http://127.0.0.1:8000`. You can view the interactive API documentation at `http://127.0.0.1:8000/docs`.

### 4. Launch the Frontend
Because it is a simple HTML/JS application, you do not need Node.js or a complex web server.
Simply open `frontend/index.html` in your web browser. 

Alternatively, if you want to run a simple local server:
```bash
python -m http.server 8080 --directory frontend
```
Then navigate to `http://localhost:8080`.

## Features
- **Bias-Aware Modeling**: Automatically detects and mitigates Demographic Parity differences.
- **Optimized for Risk**: Uses F0.5 scoring on Precision-Recall curves to severely penalize false positives (approving bad loans).
- **Explainable**: The UI surfaces the top 3 driving factors (via SHAP) for every individual prediction.
