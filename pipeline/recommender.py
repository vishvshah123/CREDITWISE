"""
CreditWise - Recommender: Generate actionable improvement recommendations
for rejected loan applicants based on SHAP explanations.

WHY THIS MODULE?
Telling someone "rejected" is useless. Telling them "your DTI is 52% —
pay off your car loan to bring it under 43%" is ACTIONABLE and EMPOWERING.
"""
import numpy as np


# Recommendation templates mapping features to actionable advice
RECOMMENDATION_MAP = {
    'credit_score': {
        'display_name': 'Credit Score',
        'action': 'Improve your credit score',
        'details': [
            'Pay all bills on time for 6+ months (payment history is 35% of score)',
            'Reduce credit card balances below 30% of limits',
            'Avoid opening new credit accounts in the short term',
            'Check your credit report for errors and dispute any inaccuracies'
        ],
        'target': 'Aim for 700+ (currently: {current})',
        'timeline': '3-12 months',
        'impact': 'HIGH — credit score is the #1 factor in loan decisions'
    },
    'debt_to_income_ratio': {
        'display_name': 'Debt-to-Income Ratio',
        'action': 'Reduce your debt-to-income ratio',
        'details': [
            'Pay off smallest existing loans first (debt snowball method)',
            'Consider consolidating high-interest debts',
            'Avoid taking on new debt before re-applying',
            'Negotiate lower interest rates with current lenders'
        ],
        'target': 'Aim for DTI below 36% (ideal) or 43% (maximum)',
        'timeline': '3-6 months',
        'impact': 'HIGH — DTI is the #2 factor after credit score'
    },
    'income': {
        'display_name': 'Income',
        'action': 'Increase your documented income',
        'details': [
            'Include all income sources (side jobs, freelance, investments)',
            'If self-employed, ensure tax returns reflect actual earnings',
            'Consider a co-applicant with additional income',
            'Request a raise or promotion documentation from employer'
        ],
        'target': 'Higher income directly improves DTI and approval chances',
        'timeline': '1-6 months',
        'impact': 'MEDIUM-HIGH — directly affects affordability assessment'
    },
    'savings_balance': {
        'display_name': 'Savings',
        'action': 'Build your savings reserves',
        'details': [
            'Aim for 3-6 months of expenses as emergency fund',
            'Set up automatic savings transfers each payday',
            'Having savings shows financial discipline to lenders',
            'Consider high-yield savings accounts to grow reserves faster'
        ],
        'target': 'Savings should cover at least 10% of loan amount',
        'timeline': '3-12 months',
        'impact': 'MEDIUM — shows financial stability and discipline'
    },
    'savings_to_loan_ratio': {
        'display_name': 'Savings-to-Loan Ratio',
        'action': 'Improve your savings relative to loan amount',
        'details': [
            'Save more before applying for a large loan',
            'Consider requesting a smaller loan amount',
            'Build a larger down payment to reduce the loan needed',
            'Having reserves reassures lenders about repayment ability'
        ],
        'target': 'Aim for savings > 20% of loan amount',
        'timeline': '6-12 months',
        'impact': 'MEDIUM — demonstrates financial cushion'
    },
    'employment_years': {
        'display_name': 'Employment Stability',
        'action': 'Strengthen your employment history',
        'details': [
            'Stay at your current job for at least 2 years before re-applying',
            'Avoid job changes during the loan application process',
            'Document any promotions or salary increases',
            'If new to a field, highlight transferable experience'
        ],
        'target': 'Lenders prefer 2+ years at current employer',
        'timeline': '6-24 months',
        'impact': 'MEDIUM — stability reduces perceived risk'
    },
    'existing_loans': {
        'display_name': 'Existing Debt',
        'action': 'Reduce your number of existing loans',
        'details': [
            'Pay off at least one existing loan before re-applying',
            'Focus on the smallest balance first for quickest win',
            'Consolidate multiple small loans into one',
            'Close unused credit lines (but keep oldest accounts open)'
        ],
        'target': 'Fewer active loans = lower perceived risk',
        'timeline': '3-12 months',
        'impact': 'MEDIUM — fewer obligations improve debt profile'
    },
    'previous_defaults': {
        'display_name': 'Previous Defaults',
        'action': 'Address your default history',
        'details': [
            'Defaults stay on credit reports for 7 years — time heals',
            'Settle any outstanding defaulted accounts',
            'Write a letter of explanation to lenders about past difficulties',
            'Build a positive payment track record going forward'
        ],
        'target': 'Zero defaults is ideal; show recovery pattern',
        'timeline': '12-24 months for meaningful improvement',
        'impact': 'VERY HIGH — defaults are the strongest negative signal'
    },
    'loan_amount': {
        'display_name': 'Loan Amount',
        'action': 'Consider requesting a smaller loan',
        'details': [
            'A smaller loan has lower monthly payments and better DTI',
            'Consider making a larger down payment',
            'Split the need into a smaller initial loan with future refinancing',
            'Ensure the loan amount matches property value for secured loans'
        ],
        'target': 'Lower loan amount = higher approval probability',
        'timeline': 'Immediate — adjustable at application time',
        'impact': 'MEDIUM — directly improves DTI and LTV ratios'
    },
    'has_co_applicant': {
        'display_name': 'Co-Applicant',
        'action': 'Apply with a co-applicant',
        'details': [
            'A co-applicant with stable income significantly boosts approval',
            'Spouse, parent, or business partner can be co-applicants',
            'Co-applicant shares responsibility — choose someone reliable',
            'Their credit score and income are also evaluated'
        ],
        'target': 'Adding a co-applicant can improve chances by 20-40%',
        'timeline': 'Immediate',
        'impact': 'HIGH — reduces lender risk substantially'
    },
    'loan_to_value_ratio': {
        'display_name': 'Loan-to-Value Ratio',
        'action': 'Improve your collateral position',
        'details': [
            'Make a larger down payment to reduce the loan-to-value ratio',
            'Consider a less expensive property that better matches your budget',
            'Get the property professionally appraised for maximum value',
            'LTV below 80% avoids private mortgage insurance (PMI)'
        ],
        'target': 'Aim for LTV below 80%',
        'timeline': 'Immediate (save for larger down payment)',
        'impact': 'MEDIUM-HIGH for secured loans'
    },
    'income_stability_score': {
        'display_name': 'Income Stability',
        'action': 'Demonstrate income stability',
        'details': [
            'Maintain consistent employment at one employer',
            'If self-employed, show 2+ years of stable/growing revenue',
            'Provide additional documentation of income sources',
            'Avoid career changes before applying'
        ],
        'target': 'Longer employment + steady income = higher stability score',
        'timeline': '6-24 months',
        'impact': 'MEDIUM — stability signals reliable repayment'
    },
    'risk_score_composite': {
        'display_name': 'Overall Risk Profile',
        'action': 'Improve your overall financial profile',
        'details': [
            'Focus on the top 2-3 weakest areas first',
            'Small improvements across multiple factors compound',
            'Consider financial counseling for a personalized plan',
            'Track your progress monthly'
        ],
        'target': 'Balanced improvement across credit, debt, and savings',
        'timeline': '6-12 months',
        'impact': 'HIGH — composite improvements have multiplicative effects'
    },
    'credit_utilization': {
        'display_name': 'Credit Utilization',
        'action': 'Lower your credit utilization',
        'details': [
            'Keep credit card balances below 30% of limits',
            'Pay down revolving debt before applying',
            'Request credit limit increases (without new spending)',
            'Make multiple payments per month to keep balances low'
        ],
        'target': 'Utilization below 30% (ideal: below 10%)',
        'timeline': '1-3 months',
        'impact': 'HIGH — directly affects credit score'
    },
    'expense_to_income_ratio': {
        'display_name': 'Expense-to-Income Ratio',
        'action': 'Reduce your monthly expenses relative to income',
        'details': [
            'Review and cut non-essential subscriptions and expenses',
            'Negotiate lower rates on insurance, utilities, and phone plans',
            'Consider downsizing housing if rent is too high',
            'Increase income through side work or career advancement'
        ],
        'target': 'Keep expenses below 60% of monthly income',
        'timeline': '1-3 months',
        'impact': 'MEDIUM — frees up cash for loan payments'
    },
    'disposable_income': {
        'display_name': 'Disposable Income',
        'action': 'Increase your available cash flow',
        'details': [
            'Reduce monthly expenses to free up disposable income',
            'Pay off existing debts to reduce monthly obligations',
            'Consider a longer loan term to reduce monthly payments',
            'Increase income through additional work or negotiation'
        ],
        'target': 'Positive disposable income after all payments',
        'timeline': '1-6 months',
        'impact': 'HIGH — negative disposable income = automatic rejection'
    }
}


class Recommender:
    """Generate personalized improvement recommendations for rejected applicants."""
    
    def __init__(self, feature_names, scaler=None):
        self.feature_names = feature_names
        self.scaler = scaler
    
    def generate_recommendations(self, instance, shap_values, prediction, 
                                  probability, top_n=5):
        """
        Generate actionable recommendations based on SHAP analysis.
        
        HOW IT WORKS:
        1. Get SHAP values (which features hurt this person most)
        2. Sort by negative impact (features pushing toward rejection)
        3. Map each negative feature to a specific, actionable recommendation
        4. Prioritize by impact magnitude (fix the biggest problems first)
        
        RETURNS: List of recommendations with:
        - What's wrong (current value vs ideal)
        - What to do (specific actions)
        - Expected impact (how much it could help)
        - Timeline (how long improvement takes)
        """
        if prediction == 1:
            return {
                'status': 'APPROVED',
                'message': 'Congratulations! Your loan has been approved.',
                'probability': float(probability),
                'recommendations': [],
                'positive_factors': self._get_positive_factors(shap_values, instance)
            }
        
        # Get features with NEGATIVE SHAP values (pushing toward rejection)
        contributions = []
        if hasattr(instance, 'values'):
            values = instance.values.flatten()
        else:
            values = np.array(instance).flatten()
        
        for feat, val, sv in zip(self.feature_names, values, shap_values):
            if sv < 0:  # Negative = pushing toward rejection
                contributions.append({
                    'feature': feat,
                    'value': float(val),
                    'shap_value': float(sv),
                    'abs_impact': abs(float(sv))
                })
        
        # Sort by impact (biggest problems first)
        contributions.sort(key=lambda x: x['abs_impact'], reverse=True)
        
        # Generate recommendations for top-N negative factors
        recommendations = []
        for contrib in contributions[:top_n]:
            feat = contrib['feature']
            
            # Find matching recommendation template
            rec_template = None
            for key in RECOMMENDATION_MAP:
                if key in feat or feat in key:
                    rec_template = RECOMMENDATION_MAP[key]
                    break
            
            if rec_template is None:
                # Generic recommendation for unmapped features
                rec_template = {
                    'display_name': feat.replace('_', ' ').title(),
                    'action': f'Improve your {feat.replace("_", " ")}',
                    'details': ['Consult a financial advisor for personalized guidance'],
                    'target': 'Improvement in this area would help your application',
                    'timeline': 'Varies',
                    'impact': 'Contributes to overall profile strength'
                }
            
            rec = {
                'priority': len(recommendations) + 1,
                'feature': feat,
                'display_name': rec_template['display_name'],
                'current_impact': f"This factor reduced your approval chance by {contrib['abs_impact']:.1%}",
                'action': rec_template['action'],
                'steps': rec_template['details'],
                'target': rec_template['target'].format(current=f"{contrib['value']:.0f}"),
                'timeline': rec_template['timeline'],
                'impact_level': rec_template['impact'],
                'shap_value': contrib['shap_value']
            }
            recommendations.append(rec)
        
        # Also get positive factors (what's working well)
        positive_factors = self._get_positive_factors(shap_values, instance)
        
        return {
            'status': 'REJECTED',
            'message': 'Your loan application was not approved at this time.',
            'probability': float(probability),
            'recommendations': recommendations,
            'positive_factors': positive_factors,
            'summary': self._generate_summary(recommendations)
        }
    
    def _get_positive_factors(self, shap_values, instance):
        """Identify what's working in the applicant's favor."""
        if hasattr(instance, 'values'):
            values = instance.values.flatten()
        else:
            values = np.array(instance).flatten()
        
        positives = []
        for feat, val, sv in zip(self.feature_names, values, shap_values):
            if sv > 0.01:  # Meaningful positive contribution
                display = feat.replace('_', ' ').title()
                positives.append({
                    'feature': feat,
                    'display_name': display,
                    'impact': f"+{sv:.1%} toward approval"
                })
        
        positives.sort(key=lambda x: float(x['impact'].strip('+% toward approval').replace('%',''))/100 
                       if 'toward' in x['impact'] else 0, reverse=True)
        return positives[:5]
    
    def _generate_summary(self, recommendations):
        """Generate a plain-English summary of what to improve."""
        if not recommendations:
            return "Your profile is strong overall."
        
        top = recommendations[0]
        summary = f"Your biggest improvement opportunity is your {top['display_name'].lower()}. "
        summary += f"{top['action']}. "
        
        if len(recommendations) > 1:
            others = [r['display_name'].lower() for r in recommendations[1:3]]
            summary += f"Also focus on improving your {' and '.join(others)}. "
        
        summary += f"With these changes, you could significantly improve your approval chances."
        return summary
