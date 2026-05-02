"""
CreditWise Loan Approval System — Streamlit Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="CreditWise — Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e, #16213e); }
.stMetric { background: rgba(255,255,255,0.05); border-radius: 12px; padding: 16px;
            border: 1px solid rgba(255,255,255,0.1); }
h1, h2, h3 { color: #e0e0ff !important; }
.approved-card { background: linear-gradient(135deg, #00b09b, #96c93d); border-radius: 16px;
                 padding: 24px; color: white; text-align: center; }
.rejected-card { background: linear-gradient(135deg, #eb3349, #f45c43); border-radius: 16px;
                 padding: 24px; color: white; text-align: center; }
.rec-card { background: rgba(255,255,255,0.06); border-radius: 12px; padding: 16px;
            border-left: 4px solid #667eea; margin: 8px 0; }
.bias-good { color: #00e676; font-weight: 600; }
.bias-bad { color: #ff5252; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    """Load trained model and pipeline data."""
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl')
    pipeline_path = os.path.join(os.path.dirname(__file__), 'models', 'pipeline_data.pkl')
    if not os.path.exists(model_path):
        return None, None
    model_data = joblib.load(model_path)
    pipeline_data = joblib.load(pipeline_path)
    return model_data, pipeline_data


def render_sidebar():
    """Render sidebar navigation."""
    st.sidebar.markdown("# 🏦 CreditWise")
    st.sidebar.markdown("*Fair & Explainable Lending*")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigate", [
        "🏠 Overview",
        "🔮 Predict Loan",
        "📊 Model Performance",
        "⚖️ Bias Audit",
        "🧠 Explainability"
    ])
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Pipeline**: Dataset → Bias Detect → Neutralize → Train → Explain")
    return page


def render_overview(model_data, pipeline_data):
    """Render the overview page."""
    st.markdown("# 🏦 CreditWise Loan Approval System")
    st.markdown("### End-to-End ML Pipeline with Bias Detection & Explainable AI")
    
    col1, col2, col3, col4 = st.columns(4)
    if pipeline_data:
        results = pipeline_data.get('model_results')
        if results is not None and len(results) > 0:
            best = results.loc[results['AUC-ROC'].idxmax()]
            col1.metric("Best Model", model_data['model_name'] if model_data else "N/A")
            col2.metric("AUC-ROC", f"{best['AUC-ROC']:.4f}")
            col3.metric("Precision", f"{best['Precision']:.4f}")
            col4.metric("Optimal Threshold", f"{pipeline_data.get('optimal_threshold', 0.5):.4f}")
    
    st.markdown("---")
    st.markdown("### 🔄 Pipeline Architecture")
    st.markdown("""
    ```
    📊 Dataset Generation (5000 synthetic applications with known biases)
         ↓
    🔍 Bias Detection (Demographic Parity, Disparate Impact, Chi-Square)
         ↓
    ⚖️ Bias Neutralization (Reweighting, Proxy Detection)
         ↓
    🛠️ Feature Engineering (DTI, LTV, Risk Score — 8 domain features)
         ↓
    🤖 Model Training (LogReg, RF, XGBoost, GBM — with sample weights)
         ↓
    🎯 Threshold Optimization (F0.5 — precision-weighted)
         ↓
    🧠 Explainable AI (SHAP + LIME for every prediction)
         ↓
    💡 Recommendations (Actionable steps for rejected applicants)
    ```
    """)
    
    st.markdown("### 📋 Step-by-Step Reasoning")
    with st.expander("Why Synthetic Data?"):
        st.markdown("""
        - We inject **known biases** (gender, age) so bias detection can be validated
        - Control ground truth relationship between features and approval
        - No privacy/licensing concerns with real financial data
        """)
    with st.expander("Why These Features?"):
        st.markdown("""
        | Feature | Reasoning |
        |---------|-----------|
        | Credit Score | Direct creditworthiness — #1 factor in real lending |
        | Income | Primary ability-to-repay indicator |
        | DTI Ratio | Banks reject > 43%. Measures repayment burden |
        | LTV Ratio | Collateral coverage for secured loans |
        | Previous Defaults | Strongest predictor of future default |
        | Employment Years | Job stability = repayment reliability |
        """)
    with st.expander("Why Bias Reduction Before Training?"):
        st.markdown("""
        - ML models **inherit biases** from historical data
        - If past data approved fewer females, model learns to discriminate
        - We detect bias FIRST, then neutralize via **sample reweighting**
        - The model sees an "unbiased" version of reality during training
        """)


def render_predict(model_data, pipeline_data):
    """Render the prediction page with input form."""
    st.markdown("# 🔮 Loan Approval Prediction")
    st.markdown("Enter applicant details to get an instant decision with full explanation.")
    
    if not model_data or not pipeline_data:
        st.error("⚠️ Model not found. Run `python train_pipeline.py` first.")
        return
    
    model = model_data['model']
    preprocessor = pipeline_data['preprocessor']
    feature_names = pipeline_data['feature_names']
    threshold = pipeline_data.get('optimal_threshold', 0.5)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Personal Info")
        age = st.slider("Age", 21, 65, 35)
        education = st.selectbox("Education", ['High School', 'Bachelor', 'Master', 'PhD'])
        marital = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
        dependents = st.number_input("Dependents", 0, 5, 1)
    
    with col2:
        st.markdown("#### 💰 Financial Info")
        income = st.number_input("Annual Income ($)", 15000, 500000, 55000, step=5000)
        emp_type = st.selectbox("Employment Type", ['Salaried', 'Self-Employed', 'Freelancer', 'Business Owner'])
        emp_years = st.slider("Employment Years", 0, 40, 5)
        credit_score = st.slider("Credit Score", 300, 850, 680)
        existing_loans = st.number_input("Existing Loans", 0, 8, 1)
        previous_defaults = st.number_input("Previous Defaults", 0, 3, 0)
    
    with col3:
        st.markdown("#### 🏦 Loan Details")
        loan_amount = st.number_input("Loan Amount ($)", 5000, 1000000, 100000, step=5000)
        loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 120, 180, 240, 360])
        loan_purpose = st.selectbox("Loan Purpose", ['Home', 'Car', 'Education', 'Personal', 'Business'])
        property_value = st.number_input("Property Value ($)", 0, 2000000, 150000, step=10000)
        monthly_expenses = st.number_input("Monthly Expenses ($)", 500, 20000, 2500, step=500)
        savings = st.number_input("Savings ($)", 0, 1000000, 15000, step=1000)
        co_applicant = st.checkbox("Has Co-Applicant")
    
    if st.button("🔍 Analyze Application", use_container_width=True):
        _run_prediction(model, preprocessor, feature_names, threshold, pipeline_data,
                       age, education, marital, dependents, income, emp_type, emp_years,
                       credit_score, existing_loans, previous_defaults, loan_amount,
                       loan_term, loan_purpose, property_value, monthly_expenses,
                       savings, co_applicant)


def _run_prediction(model, preprocessor, feature_names, threshold, pipeline_data,
                    age, education, marital, dependents, income, emp_type, emp_years,
                    credit_score, existing_loans, previous_defaults, loan_amount,
                    loan_term, loan_purpose, property_value, monthly_expenses,
                    savings, co_applicant):
    """Process a single prediction with explanations and recommendations."""
    from pipeline.feature_engineer import FeatureEngineer
    from pipeline.explainer import Explainer
    from pipeline.recommender import Recommender
    
    # Build input dataframe
    input_data = pd.DataFrame([{
        'age': age, 'gender': 'Unknown', 'education': education,
        'marital_status': marital, 'num_dependents': dependents,
        'income': float(income), 'employment_type': emp_type,
        'employment_years': float(emp_years), 'credit_score': float(credit_score),
        'existing_loans': existing_loans, 'monthly_expenses': float(monthly_expenses),
        'savings_balance': float(savings), 'previous_defaults': previous_defaults,
        'has_co_applicant': int(co_applicant), 'loan_amount': float(loan_amount),
        'loan_term_months': loan_term, 'property_value': float(property_value),
        'loan_purpose': loan_purpose, 'loan_approved': 0
    }])
    
    # Feature engineering
    engineer = FeatureEngineer()
    input_data = engineer.engineer_features(input_data)
    
    # Preprocessing (encode, scale) — reuse fitted preprocessor
    input_processed = preprocessor.handle_missing_values(input_data)
    input_processed = preprocessor.encode_categorical(input_processed, fit=False)
    
    protected_to_remove = ['gender']
    for col in protected_to_remove:
        if col in input_processed.columns:
            input_processed.drop(columns=[col], inplace=True)
    if 'loan_approved' in input_processed.columns:
        input_processed.drop(columns=['loan_approved'], inplace=True)
    
    # Align columns
    for col in feature_names:
        if col not in input_processed.columns:
            input_processed[col] = 0
    input_processed = input_processed[feature_names]
    
    # Scale
    input_scaled = input_processed.copy()
    scale_cols = [c for c in preprocessor.numeric_cols if c in input_scaled.columns]
    if scale_cols:
        input_scaled[scale_cols] = preprocessor.scaler.transform(input_scaled[scale_cols])
    
    # Predict
    proba = model.predict_proba(input_scaled)[:, 1][0]
    prediction = 1 if proba >= threshold else 0
    
    # Display result
    st.markdown("---")
    if prediction == 1:
        st.markdown(f"""
        <div class="approved-card">
            <h1>✅ LOAN APPROVED</h1>
            <h3>Confidence: {proba:.1%}</h3>
            <p>Threshold: {threshold:.4f} | Score: {proba:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="rejected-card">
            <h1>❌ LOAN REJECTED</h1>
            <h3>Confidence: {1-proba:.1%}</h3>
            <p>Threshold: {threshold:.4f} | Score: {proba:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # SHAP explanation
    st.markdown("### 🧠 Why This Decision? (SHAP Analysis)")
    try:
        import shap
        explainer_obj = Explainer(model, input_scaled, feature_names)
        shap_vals = explainer_obj.explain_shap(input_scaled)
        contributions = explainer_obj.get_feature_contributions(
            input_scaled.iloc[0], shap_vals[0]
        )
        
        # Waterfall-style bar chart
        top_contribs = contributions[:10]
        features_list = [c['feature'].replace('_', ' ').title()[:25] for c in top_contribs]
        values_list = [c['shap_value'] for c in top_contribs]
        colors = ['#00e676' if v > 0 else '#ff5252' for v in values_list]
        
        fig = go.Figure(go.Bar(
            x=values_list, y=features_list, orientation='h',
            marker_color=colors,
            text=[f"{v:+.4f}" for v in values_list],
            textposition='outside'
        ))
        fig.update_layout(
            title="Feature Contributions to Decision",
            xaxis_title="SHAP Value (+ = Toward Approval, - = Toward Rejection)",
            yaxis=dict(autorange="reversed"),
            height=400, template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # LIME explanation
        st.markdown("### 🍋 LIME Explanation (Cross-Validation)")
        try:
            lime_exp = explainer_obj.explain_lime_single(input_scaled.iloc[0], num_features=8)
            lime_data = lime_exp.as_list()
            lime_df = pd.DataFrame(lime_data, columns=['Feature Rule', 'Weight'])
            lime_df['Direction'] = lime_df['Weight'].apply(
                lambda w: '✅ Toward Approval' if w > 0 else '❌ Toward Rejection')
            st.dataframe(lime_df, use_container_width=True)
        except Exception as e:
            st.warning(f"LIME explanation unavailable: {e}")
        
        # Recommendations for rejected applicants
        if prediction == 0:
            st.markdown("---")
            st.markdown("## 💡 How to Improve Your Profile")
            st.markdown("*Based on AI analysis of your application, here are personalized recommendations:*")
            
            recommender = Recommender(feature_names)
            recs = recommender.generate_recommendations(
                input_scaled.iloc[0], shap_vals[0], prediction, proba
            )
            
            if recs.get('summary'):
                st.info(f"**Summary:** {recs['summary']}")
            
            for rec in recs.get('recommendations', []):
                impact_emoji = {'HIGH': '🔴', 'VERY HIGH': '🔴', 
                               'MEDIUM-HIGH': '🟠', 'MEDIUM': '🟡'}.get(
                    rec['impact_level'].split('—')[0].strip(), '🟢')
                
                st.markdown(f"""
                <div class="rec-card">
                    <h4>{impact_emoji} Priority #{rec['priority']}: {rec['action']}</h4>
                    <p><strong>Impact:</strong> {rec['current_impact']}</p>
                    <p><strong>Target:</strong> {rec['target']}</p>
                    <p><strong>Timeline:</strong> {rec['timeline']} | 
                       <strong>Impact Level:</strong> {rec['impact_level']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander(f"📋 Action Steps for: {rec['display_name']}"):
                    for step in rec['steps']:
                        st.markdown(f"- {step}")
            
            # Positive factors
            if recs.get('positive_factors'):
                st.markdown("### ✅ What's Working in Your Favor")
                for pf in recs['positive_factors']:
                    st.markdown(f"- **{pf['display_name']}**: {pf['impact']}")
    
    except Exception as e:
        st.error(f"Explanation error: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_model_performance(model_data, pipeline_data):
    """Render model comparison and threshold analysis."""
    st.markdown("# 📊 Model Performance")
    
    if not pipeline_data:
        st.error("Run training pipeline first.")
        return
    
    results = pipeline_data.get('model_results')
    if results is not None and len(results) > 0:
        st.markdown("### Model Comparison")
        st.dataframe(results.style.highlight_max(axis=0, subset=[
            'Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC'
        ]), use_container_width=True)
        
        # Bar chart comparison
        fig = go.Figure()
        for metric in ['Precision', 'Recall', 'F1', 'AUC-ROC']:
            fig.add_trace(go.Bar(name=metric, x=results['Model'], y=results[metric]))
        fig.update_layout(
            barmode='group', title='Model Metrics Comparison',
            template='plotly_dark', height=400,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Threshold analysis
    optimizer = pipeline_data.get('threshold_optimizer')
    if optimizer and optimizer.threshold_analysis:
        st.markdown("### 🎯 Threshold Optimization")
        st.markdown(f"**Optimal Threshold: {optimizer.optimal_threshold:.4f}** (F0.5 maximized)")
        
        th_df = pd.DataFrame(optimizer.threshold_analysis)
        st.dataframe(th_df, use_container_width=True)
        
        # PR curve data
        if optimizer.pr_curve_data:
            pr = optimizer.pr_curve_data
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=pr['recalls'], y=pr['precisions'],
                                      mode='lines', name='PR Curve',
                                      line=dict(color='#667eea', width=2)))
            fig2.update_layout(
                title='Precision-Recall Curve', xaxis_title='Recall',
                yaxis_title='Precision', template='plotly_dark', height=400,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    # Feature importance
    importance = pipeline_data.get('global_importance')
    if importance is not None:
        st.markdown("### 📈 Global Feature Importance (SHAP)")
        top_imp = importance.head(15)
        fig3 = go.Figure(go.Bar(
            x=top_imp['mean_abs_shap'], y=top_imp['feature'],
            orientation='h', marker_color='#667eea'
        ))
        fig3.update_layout(
            title='Mean |SHAP| — Which Features Matter Most?',
            yaxis=dict(autorange="reversed"), template='plotly_dark',
            height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig3, use_container_width=True)


def render_bias_audit(pipeline_data):
    """Render bias detection and reduction results."""
    st.markdown("# ⚖️ Bias Audit Report")
    st.markdown("*Comparing fairness metrics before and after bias mitigation*")
    
    if not pipeline_data:
        st.error("Run training pipeline first.")
        return
    
    pre_bias = pipeline_data.get('pre_bias_report', {})
    post_bias = pipeline_data.get('post_bias_report', {})
    
    # Pre-mitigation
    if pre_bias and 'gender' in pre_bias:
        st.markdown("### 📊 Pre-Mitigation (Raw Data)")
        pb = pre_bias['gender']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Demographic Parity Diff", f"{pb['demographic_parity_diff']:.4f}",
                    delta="Biased" if pb['is_biased'] else "Fair",
                    delta_color="inverse")
        col2.metric("Disparate Impact Ratio", f"{pb['disparate_impact_ratio']:.4f}",
                    delta="Fails 4/5 rule" if pb['disparate_impact_ratio'] < 0.8 else "Passes",
                    delta_color="inverse" if pb['disparate_impact_ratio'] < 0.8 else "normal")
        col3.metric("Chi² p-value", f"{pb['p_value']:.6f}",
                    delta="Significant bias" if pb['p_value'] < 0.05 else "No significant bias")
        
        # Approval rates
        rates = pb['approval_rates']
        fig = go.Figure(go.Bar(
            x=list(rates.keys()), y=[v * 100 for v in rates.values()],
            marker_color=['#667eea', '#f093fb'],
            text=[f"{v*100:.1f}%" for v in rates.values()],
            textposition='outside'
        ))
        fig.update_layout(
            title='Approval Rates by Gender (Pre-Mitigation)',
            yaxis_title='Approval Rate (%)', template='plotly_dark',
            height=350, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Post-mitigation
    if post_bias and 'gender' in post_bias:
        st.markdown("### ✅ Post-Mitigation (After Bias Reduction)")
        pb_post = post_bias['gender']
        
        metrics = pb_post.get('group_metrics', {})
        if metrics:
            rates_post = {g: m['approval_rate'] for g, m in metrics.items()}
            post_dpd = max(rates_post.values()) - min(rates_post.values())
            post_dir = min(rates_post.values()) / max(rates_post.values()) if max(rates_post.values()) > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("DPD (Post)", f"{post_dpd:.4f}")
            col2.metric("DIR (Post)", f"{post_dir:.4f}")
            eq_odds = pb_post.get('equalized_odds_satisfied', False)
            col3.metric("Equalized Odds", "✅ Satisfied" if eq_odds else "⚠️ Violated")
            
            # Comparison chart
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=list(rates_post.keys()), y=[v * 100 for v in rates_post.values()],
                marker_color=['#667eea', '#f093fb'],
                text=[f"{v*100:.1f}%" for v in rates_post.values()],
                textposition='outside'
            ))
            fig2.update_layout(
                title='Approval Rates by Gender (Post-Mitigation)',
                yaxis_title='Approval Rate (%)', template='plotly_dark',
                height=350, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Improvement summary
            if pre_bias and 'gender' in pre_bias:
                pre_dpd = pre_bias['gender']['demographic_parity_diff']
                improvement = (1 - post_dpd / pre_dpd) * 100 if pre_dpd > 0 else 0
                st.success(f"🎉 Bias reduced by **{improvement:.1f}%** "
                          f"(DPD: {pre_dpd:.4f} → {post_dpd:.4f})")
    
    # Methodology
    with st.expander("📖 Bias Reduction Methodology"):
        st.markdown("""
        **Our 3-step approach:**
        1. **Proxy Detection**: Identify features correlated with gender (>0.25 threshold)
        2. **Sample Reweighting**: Upweight underrepresented group-outcome combinations
        3. **Protected Attribute Removal**: Gender dropped from training features
        
        **Metrics Explained:**
        - **Demographic Parity Difference (DPD)**: |approval_rate_A - approval_rate_B|. Goal: < 0.05
        - **Disparate Impact Ratio (DIR)**: min_rate / max_rate. Goal: ≥ 0.80 (four-fifths rule)
        - **Equalized Odds**: TPR and FPR differences < 0.05 across groups
        """)


def render_explainability(pipeline_data):
    """Render SHAP/LIME explainability visualizations."""
    st.markdown("# 🧠 Explainability — SHAP & LIME")
    
    if not pipeline_data:
        st.error("Run training pipeline first.")
        return
    
    st.markdown("""
    ### Why Two Methods?
    - **SHAP** (SHapley Additive exPlanations): Mathematically grounded in game theory. 
      Provides globally consistent feature attributions.
    - **LIME** (Local Interpretable Model-agnostic Explanations): Creates local linear 
      approximations. Simpler, faster, intuitive.
    - **Cross-validation**: When both agree → high confidence in the explanation.
    """)
    
    importance = pipeline_data.get('global_importance')
    if importance is not None:
        st.markdown("### Global Feature Importance (SHAP)")
        top = importance.head(15)
        fig = go.Figure(go.Bar(
            x=top['mean_abs_shap'], y=top['feature'], orientation='h',
            marker=dict(color=top['mean_abs_shap'],
                       colorscale='Viridis', showscale=True)
        ))
        fig.update_layout(
            yaxis=dict(autorange="reversed"), template='plotly_dark',
            height=500, title='Mean |SHAP Value| — Feature Importance',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📋 Feature Importance Table")
        imp_display = importance.copy()
        imp_display['rank'] = range(1, len(imp_display) + 1)
        imp_display = imp_display[['rank', 'feature', 'mean_abs_shap']]
        imp_display.columns = ['Rank', 'Feature', 'Mean |SHAP|']
        st.dataframe(imp_display, use_container_width=True)
    
    st.markdown("---")
    st.info("💡 **Tip:** Go to the **Predict Loan** tab to see per-applicant "
            "SHAP waterfall and LIME explanations for individual predictions.")


# ── Main App ──
def main():
    model_data, pipeline_data = load_artifacts()
    page = render_sidebar()
    
    if "Overview" in page:
        render_overview(model_data, pipeline_data)
    elif "Predict" in page:
        render_predict(model_data, pipeline_data)
    elif "Model" in page:
        render_model_performance(model_data, pipeline_data)
    elif "Bias" in page:
        render_bias_audit(pipeline_data)
    elif "Explain" in page:
        render_explainability(pipeline_data)


if __name__ == '__main__':
    main()
