"""
Business Rules Engine: Hard underwriting caps applied AFTER ML scoring.
Real banks use both ML + rule-based guardrails.
"""

RULES = {
    'max_dti': 0.50,           # Debt-to-Income > 50% → auto-reject
    'max_utilization': 82.0,   # Credit Utilization > 82% → auto-reject
    'min_income_coverage': 1.2,# Monthly income must be 1.2× EMI minimum
    'min_employment': 0.5,     # Less than 6 months employment → high risk
}

def apply_rules(row: dict, prob: float) -> dict:
    """
    Applies hard banking rules on top of the ML probability.
    Returns adjusted probability and list of triggered rules.
    """
    total_income = row['ApplicantIncome'] + row['CoapplicantIncome']
    monthly_income = total_income / 12
    emi = (row['LoanAmount'] * 1000) / row['Loan_Amount_Term']
    monthly_debt = row['Existing_Debt'] * 0.03
    dti = (emi + monthly_debt) / (monthly_income + 1)
    income_coverage = monthly_income / (emi + 1)

    triggered = []
    adjusted_prob = prob

    if dti > RULES['max_dti']:
        triggered.append(f"DTI {dti:.1%} exceeds maximum 50%")
        adjusted_prob = min(adjusted_prob, 0.25)

    if row['Credit_Utilization'] > RULES['max_utilization']:
        triggered.append(f"Credit utilization {row['Credit_Utilization']:.0f}% exceeds 82%")
        adjusted_prob = min(adjusted_prob, 0.30)

    if income_coverage < RULES['min_income_coverage']:
        triggered.append(f"Income coverage {income_coverage:.2f}x below minimum 1.2x")
        adjusted_prob = min(adjusted_prob, 0.35)

    if row['Employment_Stability'] < RULES['min_employment']:
        triggered.append("Employment < 6 months — insufficient stability")
        adjusted_prob = min(adjusted_prob, 0.40)

    return {
        'adjusted_probability': round(adjusted_prob, 4),
        'rules_triggered': triggered,
        'dti': round(dti, 4),
        'income_coverage': round(income_coverage, 4),
        'emi': round(emi, 2),
    }
