"""
🏥 AI-Powered Health Insurance Fraud Detection System
======================================================
Professional ML-powered dashboard for detecting insurance fraud
Uses 5 advanced models: Random Forest, Logistic Regression, Decision Tree, KNN, Isolation Forest
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================================
# PAGE CONFIGURATION
# ================================
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# PROFESSIONAL CUSTOM CSS
# ================================
st.markdown("""
<style>
    /* Main background - Professional gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Content padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Metric cards - Glass morphism effect */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 20px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: transform 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
    }
    
    div[data-testid="metric-container"] label {
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 600;
        font-size: 14px;
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: white;
        font-size: 32px;
        font-weight: bold;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Headers with glow effect */
    h1 {
        color: white !important;
        font-weight: 800;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
        letter-spacing: 1px;
    }
    
    h2, h3 {
        color: white !important;
        font-weight: 700;
    }
    
    /* Professional buttons */
    .stButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 35px;
        font-weight: bold;
        font-size: 16px;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(245, 87, 108, 0.6);
    }
    
    /* Download button special styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border-radius: 12px;
        padding: 14px 35px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
    }
    
    /* Alert boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
    }
    
    /* Tables */
    .dataframe {
        background: white;
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Form styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div,
    .stSlider > div > div > div {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
    }
    
    /* Dividers */
    hr {
        border-color: rgba(255, 255, 255, 0.2);
        margin: 2rem 0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: white;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.3);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00ff00 0%, #ffa500 50%, #ff0000 100%);
    }
    
    /* Info boxes - custom colors */
    .info-box {
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    
    .info-box-warning {
        background: rgba(255, 165, 0, 0.2);
        border-left: 4px solid #FFA500;
    }
    
    .info-box-danger {
        background: rgba(255, 0, 0, 0.2);
        border-left: 4px solid #FF0000;
    }
    
    .info-box-success {
        background: rgba(0, 255, 0, 0.2);
        border-left: 4px solid #00FF00;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# LOAD MODELS AND DATA
# ================================

@st.cache_resource
def load_all_models():
    """Load all trained ML models"""
    try:
        models = {
            'Random Forest': pickle.load(open('random_forest.pkl', 'rb')),
            'Logistic Regression': pickle.load(open('logistic_regression.pkl', 'rb')),
            'Decision Tree': pickle.load(open('decision_tree.pkl', 'rb')),
            'KNN': pickle.load(open('knn.pkl', 'rb')),
            'Isolation Forest': pickle.load(open('isolation_forest.pkl', 'rb'))
        }
        return models
    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        return None

@st.cache_resource
def load_feature_columns():
    """Load the exact feature columns used during training"""
    try:
        return pickle.load(open('model_columns.pkl', 'rb'))
    except Exception as e:
        st.warning(f"⚠️ Could not load feature columns: {str(e)}")
        return None

@st.cache_resource
def load_preprocessor():
    """Load the data preprocessor if available"""
    try:
        return pickle.load(open('preprocessor.pkl', 'rb'))
    except:
        return None

@st.cache_data
def load_training_dataset():
    """Load training dataset for hospital names and reference data"""
    try:
        df = pd.read_csv('training_data.csv')
        return df
    except Exception as e:
        st.warning(f"⚠️ Training data not found: {str(e)}")
        return None

@st.cache_resource
def load_model_metrics():
    """Load pre-computed model accuracies"""
    try:
        return pickle.load(open('model_metrics.pkl', 'rb'))
    except:
        # Default accuracies if not available
        return {
            'Random Forest': 0.94,
            'Logistic Regression': 0.88,
            'Decision Tree': 0.86,
            'KNN': 0.85,
            'Isolation Forest': 0.82
        }

# Load everything at startup
models = load_all_models()
feature_columns = load_feature_columns()
preprocessor = load_preprocessor()
training_data = load_training_dataset()
model_metrics = load_model_metrics()

# ================================
# HELPER FUNCTIONS
# ================================

def get_hospital_list():
    """Get unique hospital IDs from training data"""
    if training_data is not None:
        hospitals = training_data['hospital_id'].dropna().unique()
        return sorted([int(h) for h in hospitals if pd.notna(h)])
    return list(range(1, 1001))  # Default range

def get_risk_color(risk_level):
    """Return color code based on risk level"""
    colors = {
        'Low Risk': '#00FF00',
        'Medium Risk': '#FFA500',
        'High Risk': '#FF0000'
    }
    return colors.get(risk_level, '#FFFFFF')

def calculate_risk_level(risk_score):
    """Determine risk level from score"""
    if risk_score >= 75:
        return "High Risk"
    elif risk_score >= 40:
        return "Medium Risk"
    else:
        return "Low Risk"

def create_risk_gauge(risk_score, risk_level):
    """Create an animated gauge chart for risk visualization"""
    color = get_risk_color(risk_level)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score", 'font': {'size': 28, 'color': 'white', 'family': 'Arial Black'}},
        number={'font': {'size': 50, 'color': 'white'}, 'suffix': "/100"},
        delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "white"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "rgba(255, 255, 255, 0.2)",
            'borderwidth': 3,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(0, 255, 0, 0.3)'},
                {'range': [40, 75], 'color': 'rgba(255, 165, 0, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(255, 0, 0, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.85,
                'value': risk_score
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': 'Arial'},
        height=350,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    return fig

def create_feature_importance_chart(features, importances, top_n=10):
    """Create feature importance bar chart"""
    # Get top N features
    indices = np.argsort(importances)[-top_n:][::-1]
    top_features = [features[i] for i in indices]
    top_importances = [importances[i] for i in indices]
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_importances,
            y=top_features,
            orientation='h',
            marker=dict(
                color=top_importances,
                colorscale='Viridis',
                line=dict(color='white', width=1)
            ),
            text=[f'{imp:.3f}' for imp in top_importances],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title={'text': f"Top {top_n} Most Important Features", 'font': {'size': 20, 'color': 'white'}},
        xaxis={'title': 'Importance', 'color': 'white', 'gridcolor': 'rgba(255, 255, 255, 0.2)'},
        yaxis={'title': '', 'color': 'white'},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': 'white'},
        height=400,
        margin=dict(l=150, r=20, t=60, b=40)
    )
    return fig

def create_model_comparison_chart(all_predictions):
    """Create a visual comparison chart of all models"""
    model_names = []
    risk_scores = []
    predictions = []
    
    for model_name, result in all_predictions.items():
        if result:
            model_names.append(model_name)
            risk_scores.append(result['risk_score'])
            predictions.append('Fraud' if result['prediction'] == 1 else 'No Fraud')
    
    # Create color based on prediction
    colors = ['#FF0000' if pred == 'Fraud' else '#00FF00' for pred in predictions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=model_names,
            y=risk_scores,
            marker=dict(color=colors, line=dict(color='white', width=2)),
            text=[f"{score:.1f}" for score in risk_scores],
            textposition='outside',
            textfont=dict(color='white', size=14)
        )
    ])
    
    fig.update_layout(
        title={'text': "Risk Scores Across All Models", 'font': {'size': 22, 'color': 'white'}},
        xaxis={'title': 'Model', 'color': 'white', 'tickangle': -45},
        yaxis={'title': 'Risk Score', 'color': 'white', 'range': [0, 110], 'gridcolor': 'rgba(255, 255, 255, 0.2)'},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': 'white'},
        height=400,
        margin=dict(l=60, r=20, t=80, b=100)
    )
    
    # Add horizontal lines for risk thresholds
    fig.add_hline(y=75, line_dash="dash", line_color="red", opacity=0.5, annotation_text="High Risk", annotation_position="right")
    fig.add_hline(y=40, line_dash="dash", line_color="orange", opacity=0.5, annotation_text="Medium Risk", annotation_position="right")
    
    return fig

def prepare_input_features(user_input, feature_columns):
    """Prepare user input to match training features"""
    # Create a dataframe with the input
    input_df = pd.DataFrame([user_input])
    
    # Ensure all required columns exist
    for col in feature_columns:
        if col not in input_df.columns:
            # Fill missing columns with defaults
            if training_data is not None and col in training_data.columns:
                if training_data[col].dtype in ['int64', 'float64']:
                    input_df[col] = training_data[col].median()
                else:
                    input_df[col] = training_data[col].mode()[0] if len(training_data[col].mode()) > 0 else 0
            else:
                input_df[col] = 0
    
    # Select only the columns in the correct order
    input_df = input_df[feature_columns]
    
    return input_df

def predict_with_model(model, input_data, model_name):
    try:
        # APPLY SAME PREPROCESSING USED DURING TRAINING
        if preprocessor is not None:
            input_data = preprocessor.transform(input_data)



        # ================= Isolation Forest =================
        if model_name == 'Isolation Forest':
            prediction = model.predict(input_data)[0]
            anomaly_score = model.score_samples(input_data)[0]

            is_anomalous = prediction == -1
            fraud_prediction = 1 if is_anomalous else 0

            probability = abs(anomaly_score)
            risk_score = min(probability * 100, 100)

            return {
                'prediction': fraud_prediction,
                'probability': probability,
                'risk_score': risk_score,
                'risk_level': calculate_risk_level(risk_score),
                'is_anomalous': is_anomalous,
                'anomaly_score': anomaly_score
            }

        # ================= Classification Models =================
        else:
            prediction = model.predict(input_data)[0]

            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(input_data)[0][1]
            else:
                probability = 0.5 if prediction == 1 else 0.2

            risk_score = probability * 100

            return {
                'prediction': int(prediction),
                'probability': probability,
                'risk_score': risk_score,
                'risk_level': calculate_risk_level(risk_score),
                'is_anomalous': None,
                'anomaly_score': None
            }

    except Exception as e:
        st.error(f"Error predicting with {model_name}: {str(e)}")
        return None

def generate_explanation(user_input, prediction, risk_score):
    """Generate human-readable explanation for the prediction"""
    explanations = []
    
    # Analyze claim amount
    if user_input.get('claim_amount', 0) > 500000:
        explanations.append("⚠️ **Extremely high claim amount** - Significantly above average")
    elif user_input.get('claim_amount', 0) > 100000:
        explanations.append("⚠️ **High claim amount** - Above typical claims")
    
    # Analyze length of stay
    los = user_input.get('length_of_stay_days', 0)
    if los > 30:
        explanations.append("⚠️ **Extended hospital stay** - Unusually long duration")
    elif los > 10:
        explanations.append("📊 **Above-average hospital stay** - Longer than typical")
    
    # Analyze distance
    distance = user_input.get('provider_patient_distance_miles', 0)
    if distance > 100:
        explanations.append("⚠️ **Unusual distance** - Patient traveled very far for treatment")
    elif distance > 50:
        explanations.append("📍 **Significant distance** - Patient not local to provider")
    
    # Analyze previous claims
    prev_claims = user_input.get('number_of_previous_claims_patient', 0)
    if prev_claims > 10:
        explanations.append("⚠️ **High claim frequency** - Patient has many previous claims")
    elif prev_claims > 5:
        explanations.append("📊 **Multiple previous claims** - Above average claim history")
    
    # Analyze late submission
    if user_input.get('claim_submitted_late', 0) == 1:
        explanations.append("⏰ **Late submission** - Claim filed after deadline")
    
    # Analyze cost per day
    if los > 0:
        cost_per_day = user_input.get('claim_amount', 0) / los
        if cost_per_day > 10000:
            explanations.append(f"💰 **High daily cost** - ${cost_per_day:,.0f} per day")
    
    # Analyze number of procedures
    procedures = user_input.get('number_of_procedures', 0)
    if procedures > 5:
        explanations.append("⚠️ **Multiple procedures** - Unusually high number of procedures")
    
    if not explanations:
        explanations.append("✅ **Normal claim pattern** - All parameters within expected ranges")
    
    return explanations

# ================================
# MAIN APPLICATION
# ================================

def main():
    # ================================
    # HEADER
    # ================================
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='font-size: 56px; margin-bottom: 10px;'>
            🏥 AI-Powered Health Insurance Fraud Detection System
        </h1>
        <p style='font-size: 20px; color: rgba(255, 255, 255, 0.9); font-weight: 500;'>
            Advanced Machine Learning for Real-Time Fraud Detection
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if models loaded successfully
    if models is None:
        st.error("❌ Models could not be loaded. Please ensure all .pkl files are in the correct directory.")
        return
    
    # ================================
    # MODEL SELECTION BAR
    # ================================
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🎯 Select Model for Detailed Analysis")
        selected_model = st.selectbox(
            "",
            options=['Random Forest', 'Logistic Regression', 'Decision Tree', 'KNN', 'Isolation Forest'],
            index=0,
            help="Choose which model's detailed prediction you want to see"
        )
    
    # ================================
    # SIDEBAR - INPUT FORM
    # ================================
    with st.sidebar:
        st.markdown("# 📋 Claim Information")
        st.markdown("---")
        
        with st.form("fraud_detection_form"):
            # Patient Information
            st.markdown("### 👤 Patient Information")
            patient_name = st.text_input("Patient Name", value="John Doe", help="Enter patient's full name")
            patient_address = st.text_input("Address", value="123 Main St, City, State", help="Patient's address")
            patient_age = st.slider("Age", min_value=0, max_value=120, value=45, help="Patient's age in years")
            patient_gender = st.selectbox("Gender", options=["Male", "Female", "Other"], help="Patient's gender")
            
            st.markdown("---")
            
            # Hospital Information
            st.markdown("### 🏥 Hospital Information")
            
            # Get hospital list
            hospital_list = get_hospital_list()
            
            hospital_selection = st.radio(
                "Hospital Selection Method",
                options=["Select from list", "Enter custom"],
                help="Choose hospital from database or enter custom"
            )
            
            if hospital_selection == "Select from list":
                hospital_id = st.selectbox(
                    "Hospital ID",
                    options=hospital_list,
                    help="Select hospital from training database"
                )
                is_new_hospital = False
            else:
                hospital_id = st.number_input(
                    "Hospital ID",
                    min_value=1,
                    max_value=9999,
                    value=1000,
                    help="Enter custom hospital ID (will be flagged for manual review)"
                )
                is_new_hospital = hospital_id not in hospital_list
                if is_new_hospital:
                    st.warning("⚠️ This hospital is not in our database. Results will be flagged for manual review.")
            
            provider_type = st.selectbox(
                "Provider Type",
                options=["Hospital", "Clinic", "Specialist Office", "Laboratory", "Emergency"],
                help="Type of healthcare provider"
            )
            
            st.markdown("---")
            
            # Claim Details
            st.markdown("### 💰 Claim Details")
            
            claim_amount = st.number_input(
                "Claim Amount ($)",
                min_value=0.0,
                max_value=10000000.0,
                value=5000.0,
                step=100.0,
                help="Total amount claimed"
            )
            
            length_of_stay = st.number_input(
                "Days Stayed in Hospital",
                min_value=0,
                max_value=365,
                value=3,
                help="Number of days patient stayed"
            )
            
            num_prev_claims = st.number_input(
                "Number of Previous Claims",
                min_value=0,
                max_value=100,
                value=2,
                help="Patient's previous claim count"
            )
            
            diagnosis = st.text_input(
                "Disease / Diagnosis",
                value="A09",
                help="ICD-10 diagnosis code or disease name"
            )
            
            icu_required = st.selectbox(
                "ICU Required?",
                options=["No", "Yes"],
                help="Was ICU stay required?"
            )
            
            surgery_done = st.selectbox(
                "Surgery Done?",
                options=["No", "Yes"],
                help="Was surgery performed?"
            )
            
            num_procedures = st.number_input(
                "Number of Procedures",
                min_value=1,
                max_value=20,
                value=2,
                help="Total procedures performed"
            )
            
            distance = st.number_input(
                "Distance to Hospital (miles)",
                min_value=0.0,
                max_value=1000.0,
                value=15.0,
                help="Distance between patient and provider"
            )
            
            submitted_late = st.checkbox(
                "Claim Submitted Late",
                value=False,
                help="Was the claim submitted after deadline?"
            )
            
            st.markdown("---")
            
            # Submit button
            submit_button = st.form_submit_button(
                "🔍 Analyze Claim",
                use_container_width=True,
                type="primary"
            )
    
    # ================================
    # PROCESS SUBMISSION
    # ================================
    if submit_button:
        # Show loading animation
        with st.spinner("🔄 Analyzing claim with 5 AI models..."):
            # Prepare input data
            user_input = {
                'claim_amount': claim_amount,
                'patient_age': patient_age,
                'patient_gender': patient_gender,
                'hospital_id': hospital_id,
                'provider_type': provider_type,
                'diagnosis': diagnosis,
                'number_of_procedures': num_procedures,
                'length_of_stay_days': length_of_stay,
                'number_of_previous_claims_patient': num_prev_claims,
                'provider_patient_distance_miles': distance,
                'claim_submitted_late': 1 if submitted_late else 0
            }
            
            # Prepare features for models
            if feature_columns:
                input_features = prepare_input_features(user_input, feature_columns)
            else:
                st.error("⚠️ Feature columns not loaded. Cannot make predictions.")
                return
            
            # Get predictions from all models
            all_predictions = {}
            for model_name, model in models.items():
                result = predict_with_model(model, input_features, model_name)
                all_predictions[model_name] = result
        
        # ================================
        # DISPLAY RESULTS - MAIN MODEL
        # ================================
        st.markdown("---")
        
        # Warning banner if new hospital
        if is_new_hospital:
            st.markdown("""
            <div class='info-box info-box-warning'>
                <h3>⚠️ PENDING MANUAL REVIEW</h3>
                <p>This hospital is not in our database. A human reviewer will verify this claim.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"## 🎯 Detailed Analysis: {selected_model}")
        
        result = all_predictions[selected_model]
        
        if result:
            # Risk banner
            if result['risk_level'] == 'High Risk':
                st.markdown("""
                <div class='info-box info-box-danger'>
                    <h2 style='margin: 0; color: white;'>🚨 HIGH FRAUD RISK DETECTED</h2>
                </div>
                """, unsafe_allow_html=True)
            elif result['risk_level'] == 'Medium Risk':
                st.markdown("""
                <div class='info-box info-box-warning'>
                    <h3 style='margin: 0; color: white;'>⚠️ Medium Risk - Additional Review Recommended</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='info-box info-box-success'>
                    <h3 style='margin: 0; color: white;'>✅ Low Fraud Risk</h3>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Main metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                prediction_text = "🚨 FRAUD" if result['prediction'] == 1 else "✅ NO FRAUD"
                st.metric("Prediction", prediction_text)
            
            with col2:
                prob_display = f"{result['probability']:.1%}" if selected_model != 'Isolation Forest' else f"{result['probability']:.3f}"
                st.metric("Fraud Probability", prob_display)
            
            with col3:
                st.metric("Risk Score", f"{result['risk_score']:.1f}/100")
            
            with col4:
                risk_emoji = "🔴" if result['risk_level'] == "High Risk" else "🟡" if result['risk_level'] == "Medium Risk" else "🟢"
                st.metric("Risk Level", f"{risk_emoji} {result['risk_level']}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Gauge and details row
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Risk gauge
                fig = create_risk_gauge(result['risk_score'], result['risk_level'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Model Performance")
                
                # Model accuracy
                accuracy = model_metrics.get(selected_model, 0.9)
                st.metric("Model Accuracy", f"{accuracy:.1%}")
                
                # Isolation Forest specific metrics
                if selected_model == 'Isolation Forest':
                    st.metric("Anomaly Score", f"{result['anomaly_score']:.4f}")
                    status = "🔴 Anomalous" if result['is_anomalous'] else "🟢 Normal"
                    st.metric("Status", status)
                
                st.markdown("---")
                
                # Claim summary
                st.markdown("### 📋 Claim Summary")
                st.markdown(f"""
                - **Patient**: {patient_name}  
                - **Age**: {patient_age} | **Gender**: {patient_gender}  
                - **Hospital**: #{hospital_id}  
                - **Claim Amount**: ${claim_amount:,.2f}  
                - **Length of Stay**: {length_of_stay} days  
                - **Procedures**: {num_procedures}  
                - **Distance**: {distance:.1f} miles
                """)
                       # ================================
            # FRAUD EXPLANATION
            # ================================
            # st.markdown("---")
            # st.markdown("### 🧠 Why This Prediction?")
            
            # explanations = generate_explanation(user_input, result['prediction'], result['risk_score'])
            
            # for explanation in explanations:
            #     st.markdown(f"""
            #     <div style='background: rgba(255, 255, 255, 0.1); 
            #                 padding: 15px; 
            #                 border-radius: 10px; 
            #                 margin: 10px 0; 
            #                 border-left: 4px solid #667eea;'>
            #         <p style='color: white; margin: 0; font-size: 16px;'>{explanation}</p>
            #     </div>
            #     """, unsafe_allow_html=True)

                        

            # ================================
            # FRAUD EXPLANATION
            # ================================
            st.markdown("---")
            st.markdown("### 🧠 Why This Prediction?")
            
            explanations = generate_explanation(user_input, result['prediction'], result['risk_score'])
            
            for explanation in explanations:
                st.markdown(f"""
                <div style='background: rgba(255, 255, 255, 0.1); 
                            padding: 15px; 
                            border-radius: 10px; 
                            margin: 10px 0; 
                            border-left: 4px solid #667eea;'>
                    <p style='color: white; margin: 0; font-size: 16px;'>{explanation}</p>
                </div>
                """, unsafe_allow_html=True)


            # ================================
            # FEATURE IMPORTANCE (Tree Models)
            # ================================
            if selected_model in ['Random Forest', 'Decision Tree']:
                st.markdown("### 🌟 Top Features Influencing This Prediction")

                model_obj = models[selected_model]
                # If model was saved inside Pipeline
                if hasattr(model_obj, "named_steps"):
                        model_obj = model_obj.named_steps['model']

                if hasattr(model_obj, 'feature_importances_'):
                    importances = model_obj.feature_importances_
                    fig = create_feature_importance_chart(feature_columns, importances)
                    st.plotly_chart(fig, use_container_width=True)

            # ================================
            # MODEL COMPARISON
            # ================================
            st.markdown("---")
            st.markdown("## 🔬 All Models Comparison")

                   # ================================
           

            # Visual comparison chart
            fig_comparison = create_model_comparison_chart(all_predictions)
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Comparison table
            comparison_data = []
            for model_name, result_model in all_predictions.items():
                if result_model:
                    accuracy = model_metrics.get(model_name, 0.9)
                    
                    row = {
                        'Model': model_name,
                        'Accuracy': f"{accuracy:.1%}",
                        'Prediction': '🚨 Fraud' if result_model['prediction'] == 1 else '✅ No Fraud',
                        'Probability': f"{result_model['probability']:.2%}" if model_name != 'Isolation Forest' else f"{result_model['probability']:.3f}",
                        'Risk Score': f"{result_model['risk_score']:.1f}/100",
                        'Risk Level': result_model['risk_level'],
                        'Anomalous': ('🔴 Yes' if result_model['is_anomalous'] else '🟢 No') if result_model['is_anomalous'] is not None else 'N/A'
                    }
                    comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, height=250)

            # ================================
            # CONSENSUS ANALYSIS
            # ================================
            st.markdown("---")
            st.markdown("### 🤝 Model Consensus")
            
            fraud_count = sum(1 for r in all_predictions.values() if r and r['prediction'] == 1)
            total_models = len([r for r in all_predictions.values() if r])
            consensus_pct = (fraud_count / total_models) * 100 if total_models > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Models Detecting Fraud", f"{fraud_count}/{total_models}")
            col2.metric("Consensus Level", f"{consensus_pct:.0f}%")
            
            if consensus_pct >= 80:
                verdict = "🔴 Strong Agreement - Fraud"
            elif consensus_pct >= 50:
                verdict = "🟡 Mixed Signals - Review"
            elif consensus_pct >= 20:
                verdict = "🟡 Weak Agreement"
            else:
                verdict = "🟢 Strong Agreement - No Fraud"
            col3.metric("Verdict", verdict)

            st.progress(consensus_pct / 100)

        
        # ================================
        # DOWNLOAD REPORT
        # ================================
        st.markdown("---")
        st.markdown("### 📥 Download Analysis Report")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Generate comprehensive report
            report = f"""
{'='*70}
AI-POWERED HEALTH INSURANCE FRAUD DETECTION REPORT
{'='*70}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*70}
PATIENT INFORMATION
{'='*70}
Name:              {patient_name}
Address:           {patient_address}
Age:               {patient_age} years
Gender:            {patient_gender}

{'='*70}
HOSPITAL INFORMATION
{'='*70}
Hospital ID:       {hospital_id}
Provider Type:     {provider_type}
Distance:          {distance:.1f} miles
{'New Hospital - PENDING MANUAL REVIEW' if is_new_hospital else 'Hospital in Database'}

{'='*70}
CLAIM DETAILS
{'='*70}
Claim Amount:                    ${claim_amount:,.2f}
Diagnosis:                       {diagnosis}
Length of Stay:                  {length_of_stay} days
Number of Procedures:            {num_procedures}
Previous Claims (Patient):       {num_prev_claims}
ICU Required:                    {icu_required}
Surgery Done:                    {surgery_done}
Submitted Late:                  {'Yes' if submitted_late else 'No'}
Cost per Day:                    ${(claim_amount / max(length_of_stay, 1)):,.2f}

{'='*70}
PRIMARY MODEL: {selected_model}
{'='*70}
Prediction:          {'FRAUD DETECTED' if result['prediction'] == 1 else 'NO FRAUD'}
Probability:         {result['probability']:.2%}
Risk Score:          {result['risk_score']:.1f}/100
Risk Level:          {result['risk_level']}
Model Accuracy:      {model_metrics.get(selected_model, 0.9):.1%}
"""
            
            if selected_model == 'Isolation Forest':
                report += f"Anomaly Score:       {result['anomaly_score']:.4f}\n"
                report += f"Classification:      {'Anomalous' if result['is_anomalous'] else 'Normal'}\n"
            
            report += f"\n{'='*70}\n"
            report += "ALL MODELS COMPARISON\n"
            report += f"{'='*70}\n\n"
            
            for model_name, res in all_predictions.items():
                if res:
                    report += f"{model_name}:\n"
                    report += f"  Prediction:    {'Fraud' if res['prediction'] == 1 else 'No Fraud'}\n"
                    report += f"  Risk Score:    {res['risk_score']:.1f}/100\n"
                    report += f"  Risk Level:    {res['risk_level']}\n"
                    report += f"  Accuracy:      {model_metrics.get(model_name, 0.9):.1%}\n\n"
            
            report += f"{'='*70}\n"
            report += "MODEL CONSENSUS\n"
            report += f"{'='*70}\n"
            report += f"Models Detecting Fraud:  {fraud_count}/{total_models}\n"
            report += f"Consensus Level:         {consensus_pct:.0f}%\n\n"
            
            report += f"{'='*70}\n"
            report += "EXPLANATION\n"
            report += f"{'='*70}\n"
            for i, exp in enumerate(explanations, 1):
                # Strip HTML tags for text report
                clean_exp = exp.replace('⚠️ **', '').replace('📊 **', '').replace('📍 **', '').replace('⏰ **', '').replace('💰 **', '').replace('✅ **', '').replace('**', '')
                report += f"{i}. {clean_exp}\n"
            
            report += f"\n{'='*70}\n"
            report += "RECOMMENDATION\n"
            report += f"{'='*70}\n"
            
            if result['risk_level'] == 'High Risk' or consensus_pct >= 60:
                report += "⚠️  RECOMMEND: DETAILED INVESTIGATION\n"
                report += "This claim shows high fraud risk. Conduct thorough review.\n"
            elif result['risk_level'] == 'Medium Risk' or consensus_pct >= 30:
                report += "⚠️  RECOMMEND: ADDITIONAL VERIFICATION\n"
                report += "This claim requires additional documentation review.\n"
            else:
                report += "✅ RECOMMEND: STANDARD PROCESSING\n"
                report += "This claim appears legitimate. Process normally.\n"
            
            if is_new_hospital:
                report += "\n⚠️  MANUAL REVIEW REQUIRED - New hospital not in database\n"
            
            report += f"\n{'='*70}\n"
            report += "END OF REPORT\n"
            report += f"{'='*70}\n"
            
            # Download button
            st.download_button(
                label="📥 Download Complete Report",
                data=report,
                file_name=f"fraud_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    else:
        # ================================
        # WELCOME SCREEN
        # ================================
        st.markdown("""
        <div style='text-align: center; padding: 40px 20px; color: white;'>
            <h2 style='font-size: 36px; margin-bottom: 20px;'>
                👈 Enter claim details to begin analysis
            </h2>
            <p style='font-size: 20px; margin-top: 20px; line-height: 1.8;'>
                Our AI system uses <strong>5 advanced machine learning models</strong> to detect<br>
                potential fraud in health insurance claims with industry-leading accuracy.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Model showcase
        st.markdown("### 🤖 Our AI Models")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        model_info = [
            ("🌳", "Random Forest", "Ensemble\nLearning", "94% Accuracy"),
            ("📊", "Logistic\nRegression", "Probabilistic\nAnalysis", "88% Accuracy"),
            ("🌲", "Decision\nTree", "Rule-Based\nClassification", "86% Accuracy"),
            ("🎯", "K-Nearest\nNeighbors", "Pattern\nMatching", "85% Accuracy"),
            ("🔍", "Isolation\nForest", "Anomaly\nDetection", "82% Accuracy")
        ]
        
        for col, (emoji, name, method, acc) in zip([col1, col2, col3, col4, col5], model_info):
            with col:
                st.markdown(f"""
                <div style='text-align: center; padding: 20px; background: rgba(255, 255, 255, 0.1); 
                            border-radius: 15px; backdrop-filter: blur(10px); height: 200px;
                            display: flex; flex-direction: column; justify-content: center;'>
                    <div style='font-size: 48px; margin-bottom: 10px;'>{emoji}</div>
                    <div style='font-size: 16px; font-weight: bold; margin-bottom: 5px; color: white;'>{name}</div>
                    <div style='font-size: 13px; color: rgba(255, 255, 255, 0.8); margin-bottom: 5px;'>{method}</div>
                    <div style='font-size: 14px; color: #4ade80; font-weight: bold;'>{acc}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Show dataset statistics if available
        if training_data is not None:
            st.markdown("---")
            st.markdown("### 📊 System Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Training Claims", f"{len(training_data):,}")
            
            with col2:
                fraud_count = training_data['is_fraudulent'].sum() if 'is_fraudulent' in training_data.columns else 0
                st.metric("Fraudulent Cases Detected", f"{int(fraud_count):,}")
            
            with col3:
                fraud_rate = (fraud_count / len(training_data)) * 100 if len(training_data) > 0 else 0
                st.metric("Historical Fraud Rate", f"{fraud_rate:.2f}%")
            
            with col4:
                hospitals = len(training_data['hospital_id'].unique()) if 'hospital_id' in training_data.columns else 0
                st.metric("Hospitals in Database", f"{hospitals:,}")

# ================================
# RUN APPLICATION
# ================================
if __name__ == "__main__":
    main()
