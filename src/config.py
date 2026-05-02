import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'real_loan_dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Dataset Columns Configuration
TARGET_COL = 'Loan_Status'
ID_COL = 'Loan_ID'

NUMERICAL_COLS = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
CATEGORICAL_COLS = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

# Protected Attributes for Bias Analysis
PROTECTED_ATTRIBUTE = 'Gender'
PRIVILEGED_GROUP = 'Male'
UNPRIVILEGED_GROUP = 'Female'
