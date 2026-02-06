"""
🏥 DEMO VERSION - Hospital Fraud Detection Dashboard
=====================================================
This version uses simulated predictions for testing the UI
Use this to test the interface before your models are ready
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import random

# Same page config and CSS as main app
st.set_page_config(
    page_title="Hospital Fraud Detection - DEMO",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# [Same CSS as main app - truncated for brevity]
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    h1, h2, h3 { color: white !important; }
    .stButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def create_risk_gauge(risk_score, risk_level):
    """Create a gauge chart for risk visualization"""
    colors = {'Low Risk': '#00FF00', 'Medium Risk': '#FFA500', 'High Risk': '#FF0000'}
    color = colors.get(risk_level, '#FFFFFF')
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score", 'font': {'size': 24, 'color': 'white'}},
        number={'font': {'size': 40, 'color': 'white'}},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(0, 255, 0, 0.3)'},
                {'range': [40, 75], 'color': 'rgba(255, 165, 0, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(255, 0, 0, 0.3)'}
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"},
        height=300
    )
    return fig

def simulate_prediction(claim_amount, num_prev_claims, distance, length_of_stay):
    """Simulate model predictions based on input features"""
    # Simple rule-based simulation
    risk_factors = 0
    
    if claim_amount > 10000:
        risk_factors += 2
    if num_prev_claims > 5:
        risk_factors += 2
    if distance > 50:
        risk_factors += 1
    if length_of_stay > 10:
        risk_factors += 1
    
    # Base probability
    base_prob = 0.2 + (risk_factors * 0.15)
    base_prob = min(base_prob, 0.95)
    
    return base_prob

def main():
    # Header
    st.markdown("""
    <h1 style='text-align: center; color: white; font-size: 48px;'>
        🏥 Hospital Fraud Detection Dashboard
    </h1>
    <p style='text-align: center; color: rgba(255,255,255,0.8); font-size: 18px;'>
        DEMO MODE - Testing Interface
    </p>
    """, unsafe_allow_html=True)
    
    st.warning("⚠️ DEMO MODE: Using simulated predictions. Load actual models for real analysis.")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Select Model")
        selected_model = st.selectbox(
            "Choose model:",
            options=['KNN', 'Decision Tree', 'Random Forest', 'Isolation Forest', 'Logistic Regression'],
            index=2
        )
        
        st.markdown("---")
        st.markdown("### 📋 Enter Claim Details")
        
        with st.form("claim_form"):
            st.markdown("#### 👤 Patient Information")
            patient_name = st.text_input("Patient Name", value="John Doe")
            patient_age = st.slider("Age", 0, 120, 45)
            patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
            st.markdown("#### 🏥 Hospital Information")
            hospital_id = st.number_input("Hospital ID", min_value=1, value=100)
            provider_type = st.selectbox("Provider Type", 
                                        ["Hospital", "Clinic", "Specialist", "Emergency"])
            
            st.markdown("#### 💰 Claim Details")
            claim_amount = st.number_input("Claim Amount ($)", min_value=0.0, value=5000.0, step=100.0)
            diagnosis = st.text_input("Diagnosis Code", value="A09")
            num_procedures = st.number_input("Number of Procedures", min_value=1, max_value=20, value=2)
            length_of_stay = st.number_input("Length of Stay (days)", min_value=0, max_value=365, value=3)
            num_prev_claims_patient = st.number_input("Patient's Previous Claims", 
                                                     min_value=0, max_value=100, value=2)
            num_prev_claims_provider = st.number_input("Provider's Previous Claims", 
                                                       min_value=0, max_value=1000, value=50)
            distance = st.number_input("Patient-Provider Distance (miles)", 
                                      min_value=0.0, max_value=1000.0, value=15.0)
            submitted_late = st.checkbox("Claim Submitted Late")
            
            st.markdown("---")
            submit_button = st.form_submit_button("🔍 Analyze Claim", use_container_width=True)
    
    if submit_button:
        # Simulate predictions for all models
        base_prob = simulate_prediction(claim_amount, num_prev_claims_patient, distance, length_of_stay)
        
        # Generate slightly different predictions for each model
        all_predictions = {}
        
        for i, model_name in enumerate(['KNN', 'Decision Tree', 'Random Forest', 'Isolation Forest', 'Logistic Regression']):
            # Add some randomness
            prob = base_prob + random.uniform(-0.1, 0.1)
            prob = max(0.05, min(0.95, prob))
            
            prediction = 1 if prob > 0.5 else 0
            risk_score = prob * 100
            
            if risk_score >= 75:
                risk_level = "High Risk"
            elif risk_score >= 40:
                risk_level = "Medium Risk"
            else:
                risk_level = "Low Risk"
            
            all_predictions[model_name] = {
                'prediction': prediction,
                'probability': prob,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'anomaly_score': -prob if model_name == 'Isolation Forest' else None,
                'is_anomalous': prob > 0.5 if model_name == 'Isolation Forest' else None
            }
        
        # Display results for selected model
        st.markdown("---")
        st.markdown(f"## 🎯 Detailed Analysis: {selected_model}")
        
        result = all_predictions[selected_model]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fraud_text = "🚨 FRAUD DETECTED" if result['prediction'] == 1 else "✅ No Fraud"
            st.metric("Prediction", fraud_text)
        
        with col2:
            st.metric("Fraud Probability", f"{result['probability']:.1%}")
        
        with col3:
            st.metric("Risk Score", f"{result['risk_score']:.1f}/100")
        
        with col4:
            st.metric("Risk Level", result['risk_level'])
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig = create_risk_gauge(result['risk_score'], result['risk_level'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Additional Metrics")
            st.metric("Model Accuracy", "92.0%")  # Demo value
            
            if selected_model == 'Isolation Forest':
                st.metric("Anomaly Score", f"{result['anomaly_score']:.4f}")
            
            st.markdown("---")
            st.markdown("**Claim Summary:**")
            st.markdown(f"- Patient: {patient_name}")
            st.markdown(f"- Age: {patient_age} | Gender: {patient_gender}")
            st.markdown(f"- Claim Amount: ${claim_amount:,.2f}")
            st.markdown(f"- Length of Stay: {length_of_stay} days")
        
        # Feature explanation
        st.markdown("---")
        st.markdown("### 🧠 Why This Prediction?")
        
        explanation_points = []
        if claim_amount > 10000:
            explanation_points.append(f"• High claim amount detected (${claim_amount:,.0f})")
        if num_prev_claims_patient > 5:
            explanation_points.append(f"• Patient has many previous claims ({num_prev_claims_patient})")
        if distance > 50:
            explanation_points.append(f"• Unusual distance ({distance:.0f} miles)")
        if submitted_late:
            explanation_points.append("• Claim was submitted late")
        if length_of_stay > 10:
            explanation_points.append(f"• Extended length of stay ({length_of_stay} days)")
        
        if not explanation_points:
            explanation_points.append("• Claim parameters appear normal")
        
        for point in explanation_points:
            st.markdown(f"<p style='color: white; font-size: 16px;'>{point}</p>", 
                       unsafe_allow_html=True)
        
        # Model comparison
        st.markdown("---")
        st.markdown("## 🔬 All Models Comparison")
        
        comparison_data = []
        for model_name, result in all_predictions.items():
            row = {
                'Model': model_name,
                'Prediction': '🚨 Fraud' if result['prediction'] == 1 else '✅ No Fraud',
                'Probability': f"{result['probability']:.2%}",
                'Risk Score': f"{result['risk_score']:.1f}/100",
                'Risk Level': result['risk_level'],
                'Accuracy': "92.0%"  # Demo value
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, height=250)
        
        # Download report
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            report = f"""
HOSPITAL FRAUD DETECTION REPORT (DEMO)
======================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PATIENT: {patient_name}
AGE: {patient_age}
CLAIM AMOUNT: ${claim_amount:,.2f}

SELECTED MODEL: {selected_model}
PREDICTION: {'FRAUD' if result['prediction'] == 1 else 'NO FRAUD'}
RISK SCORE: {result['risk_score']:.1f}/100
RISK LEVEL: {result['risk_level']}

ALL MODELS COMPARISON:
"""
            for model_name, res in all_predictions.items():
                report += f"\n{model_name}: {'Fraud' if res['prediction'] == 1 else 'No Fraud'} ({res['risk_score']:.1f}/100)"
            
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=f"fraud_report_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    else:
        st.markdown("""
        <div style='text-align: center; padding: 50px; color: white;'>
            <h2>👈 Enter claim details to test the interface</h2>
            <p style='font-size: 18px; margin-top: 20px;'>
                This demo uses simulated predictions to showcase the UI
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
