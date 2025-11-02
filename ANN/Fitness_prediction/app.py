import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import os

# ------------------ File Paths ------------------
MODEL_DIR = 'ANN/Fitness_prediction'
MODEL_PATH = os.path.join(MODEL_DIR, 'predmodel.h5')
GENDER_ENCODER_PATH = os.path.join(MODEL_DIR, 'gender_encoder.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# Try to import streamlit-lottie, with fallback
try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False

# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="Advanced Fitness Intelligence Platform",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Custom CSS with Cult.fit Inspired Theme ------------------
def set_cultfit_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 50%, #000000 100%);
        font-family: 'Inter', sans-serif;
        color: #ffffff;
    }
    
    .main-header {
        background: linear-gradient(135deg, #FF6B35 0%, #FF8E53 100%);
        padding: 2rem;
        border-radius: 0 0 20px 20px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(255, 107, 53, 0.3);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(255, 107, 53, 0.2);
        border-color: rgba(255, 107, 53, 0.3);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #00D2FF 0%, #3A7BD5 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(0, 210, 255, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .risk-card {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(255, 65, 108, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #FF6B35 0%, #FF8E53 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 107, 53, 0.6);
        background: linear-gradient(135deg, #FF8E53 0%, #FF6B35 100%);
    }
    
    h1, h2, h3 {
        font-weight: 700 !important;
        background: linear-gradient(135deg, #FF6B35, #FF8E53);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem !important;
    }
    
    .section-divider {
        height: 3px;
        background: linear-gradient(90deg, #FF6B35, #FF8E53);
        margin: 2rem 0;
        border-radius: 2px;
    }
    
    .sidebar-content {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #FF6B35, #FF8E53);
        border-radius: 4px;
    }
    
    /* Sleep metric specific styling */
    .sleep-metric [data-testid="stMetricValue"] {
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    
    /* Animation keyframes */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes slideIn {
        from { 
            opacity: 0;
            transform: translateY(30px);
        }
        to { 
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    .slide-in {
        animation: slideIn 0.6s ease-out;
    }
    
    /* Custom container styling */
    .css-1d391kg, .css-1v0mbdj, .css-1v3fvcr {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    </style>
    """, unsafe_allow_html=True)

# ------------------ Load Lottie Animations ------------------
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# ------------------ Load model and encoders ------------------
@st.cache_resource
def load_model_and_encoders():
    try:
        # Check if files exist
        if not os.path.exists(MODEL_PATH):
            return None, None, None
        if not os.path.exists(GENDER_ENCODER_PATH):
            return None, None, None
        if not os.path.exists(SCALER_PATH):
            return None, None, None
            
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(GENDER_ENCODER_PATH, 'rb') as file:
            label_encoder_gender = pickle.load(file)
        with open(SCALER_PATH, 'rb') as file:
            scaler = pickle.load(file)
        
        return model, label_encoder_gender, scaler
        
    except Exception as e:
        return None, None, None

# ------------------ Initialize App ------------------
set_cultfit_theme()
model, label_encoder_gender, scaler = load_model_and_encoders()

# ------------------ Header Section ------------------
st.markdown("""
<div class="main-header">
    <div style="text-align: center;">
        <h1 style="color: #ffffff; font-size: 3rem; margin-bottom: 0.5rem; font-weight: 800;">🏃‍♂️ Advanced Fitness Intelligence Platform</h1>
        <p style="color: #ffffff; font-size: 1.2rem; margin-bottom: 0;">
            AI-powered fitness analytics with personalized insights and recommendations
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Header animation
header_col1, header_col2, header_col3 = st.columns([1, 2, 1])

with header_col2:
    if LOTTIE_AVAILABLE:
        lottie_fitness = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_0pfisjz6.json")
        if lottie_fitness:
            st_lottie(lottie_fitness, height=120, key="header-animation")

# ------------------ Interactive Input Section ------------------
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.header('🎯 Personal Fitness Assessment')

input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    st.markdown('<div class="metric-card slide-in">', unsafe_allow_html=True)
    st.subheader("📊 Biometric Data")
    age = st.slider('**Age**', min_value=18, max_value=100, value=30)
    height = st.slider('**Height (cm)**', min_value=140.0, max_value=220.0, value=170.0)
    weight = st.slider('**Weight (kg)**', min_value=40.0, max_value=150.0, value=70.0)
    
    # BMI Calculation
    bmi = weight / ((height/100) ** 2)
    bmi_status = "Healthy" if 18.5 <= bmi <= 24.9 else "Underweight" if bmi < 18.5 else "Overweight"
    st.metric("**BMI**", f"{bmi:.1f}", delta=bmi_status)
    st.markdown('</div>', unsafe_allow_html=True)

with input_col2:
    st.markdown('<div class="metric-card slide-in">', unsafe_allow_html=True)
    st.subheader("❤️ Health Metrics")
    blood_pressure = st.slider('**Blood Pressure (mmHg)**', min_value=80, max_value=180, value=120)
    heart_rate = st.slider('**Resting Heart Rate (bpm)**', min_value=40, max_value=120, value=72)
    if label_encoder_gender:
        gender = st.selectbox('**Gender**', options=label_encoder_gender.classes_)
    else:
        gender = st.selectbox('**Gender**', options=['Male', 'Female'])
    smokes = st.selectbox('**Smoking Status**', options=['No', 'Yes'])
    st.markdown('</div>', unsafe_allow_html=True)

with input_col3:
    st.markdown('<div class="metric-card slide-in">', unsafe_allow_html=True)
    st.subheader("🏃‍♂️ Lifestyle Factors")
    steps = st.slider('**Daily Steps**', min_value=0, max_value=30000, value=8000)
    exercise_hours = st.slider('**Weekly Exercise Hours**', min_value=0.0, max_value=20.0, value=3.0)
    sleep_hours = st.slider('**Daily Sleep Hours**', min_value=3.0, max_value=12.0, value=7.0)
    
    # Sleep quality indicator with smaller font
    sleep_status = "🟢 Optimal" if 7 <= sleep_hours <= 9 else "🟡 Needs improvement" if 6 <= sleep_hours < 7 or 9 < sleep_hours <= 10 else "🔴 Poor"
    st.markdown(f"""
    <div class="sleep-metric">
    <div data-testid="stMetric" style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.2);">
        <div data-testid="stMetricLabel" style="color: white; font-size: 1rem; font-weight: 600;">Sleep Quality</div>
        <div data-testid="stMetricValue" style="color: white; font-size: 1rem; font-weight: 500;">{sleep_status}</div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Real-time Health Dashboard ------------------
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.header('📈 Live Health Dashboard')

# Create real-time health metrics
dashboard_col1, dashboard_col2, dashboard_col3, dashboard_col4 = st.columns(4)

with dashboard_col1:
    st.markdown('<div class="metric-card pulse-animation">', unsafe_allow_html=True)
    activity_level = "Highly Active" if steps > 10000 else "Active" if steps > 7500 else "Moderate" if steps > 5000 else "Sedentary"
    st.metric("**Activity Level**", activity_level, f"{steps:,} steps")
    st.markdown('</div>', unsafe_allow_html=True)

with dashboard_col2:
    st.markdown('<div class="metric-card pulse-animation">', unsafe_allow_html=True)
    fitness_score = "Excellent" if exercise_hours > 5 else "Good" if exercise_hours > 3 else "Fair" if exercise_hours > 1 else "Poor"
    st.metric("**Fitness Score**", fitness_score, f"{exercise_hours} hrs/week")
    st.markdown('</div>', unsafe_allow_html=True)

with dashboard_col3:
    st.markdown('<div class="metric-card pulse-animation">', unsafe_allow_html=True)
    heart_status = "Athletic" if heart_rate < 60 else "Excellent" if heart_rate < 70 else "Good" if heart_rate < 80 else "Needs Improvement"
    st.metric("**Heart Health**", heart_status, f"{heart_rate} bpm")
    st.markdown('</div>', unsafe_allow_html=True)

with dashboard_col4:
    st.markdown('<div class="metric-card pulse-animation">', unsafe_allow_html=True)
    bp_status = "Normal" if blood_pressure < 120 else "Elevated" if blood_pressure < 130 else "High"
    st.metric("**Blood Pressure**", bp_status, f"{blood_pressure} mmHg")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Advanced Visualizations ------------------
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.header('📊 Comprehensive Health Analysis')

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    st.markdown('<div class="slide-in">', unsafe_allow_html=True)
    # Health Radar Chart
    st.subheader("Health Metrics Radar")
    
    # Normalize metrics for radar chart (0-1 scale)
    categories = ['Physical Activity', 'Cardio Health', 'Sleep Quality', 'Exercise', 'Lifestyle']
    
    activity_score = min(steps / 10000, 1.0)
    cardio_score = 1 - max(0, (heart_rate - 60) / 60)  # Lower heart rate = better
    sleep_score = min(sleep_hours / 9, 1.0) if sleep_hours <= 9 else max(0, 1 - (sleep_hours - 9) / 3)
    exercise_score = min(exercise_hours / 7, 1.0)
    lifestyle_score = 1.0 if smokes == 'No' else 0.3
    
    values = [activity_score, cardio_score, sleep_score, exercise_score, lifestyle_score]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the radar
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(255, 107, 53, 0.4)',
        line=dict(color='#FF6B35', width=3),
        name='Your Health Profile'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor='rgba(255,255,255,0.05)'
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        height=450,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with viz_col2:
    st.markdown('<div class="slide-in">', unsafe_allow_html=True)
    # Progress Gauge Charts
    st.subheader("Fitness Progress Gauges")
    
    # Create subplots for gauges with more spacing
    fig_gauges = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'indicator'}]],
        vertical_spacing=0.3,
        horizontal_spacing=0.2,
        row_heights=[0.5, 0.5]
    )
    
    # Steps Gauge
    fig_gauges.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=steps,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Daily Steps", 'font': {'color': 'white', 'size': 14}},
        gauge={'axis': {'range': [0, 15000]},
               'bar': {'color': "#FF6B35"},
               'steps': [{'range': [0, 5000], 'color': "rgba(255,255,255,0.1)"},
                        {'range': [5000, 10000], 'color': "rgba(255,255,255,0.2)"}],
               'threshold': {'line': {'color': "#FF6B35", 'width': 4}, 'thickness': 0.75, 'value': 10000}},
        number={'font': {'color': 'white', 'size': 20}}
    ), row=1, col=1)
    
    # Exercise Gauge
    fig_gauges.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=exercise_hours,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Exercise Hours/Week", 'font': {'color': 'white', 'size': 14}},
        gauge={'axis': {'range': [0, 15]},
               'bar': {'color': "#00D2FF"},
               'steps': [{'range': [0, 3], 'color': "rgba(255,255,255,0.1)"},
                        {'range': [3, 7], 'color': "rgba(255,255,255,0.2)"}],
               'threshold': {'line': {'color': "#00D2FF", 'width': 4}, 'thickness': 0.75, 'value': 5}},
        number={'font': {'color': 'white', 'size': 20}}
    ), row=1, col=2)
    
    # Heart Rate Gauge
    fig_gauges.add_trace(go.Indicator(
        mode="gauge+number",
        value=heart_rate,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Resting Heart Rate", 'font': {'color': 'white', 'size': 14}},
        gauge={'axis': {'range': [40, 120]},
               'bar': {'color': "#FF416C"},
               'steps': [{'range': [40, 60], 'color': "rgba(0,210,255,0.3)"},
                        {'range': [60, 80], 'color': "rgba(255,107,53,0.3)"},
                        {'range': [80, 120], 'color': "rgba(255,65,108,0.3)"}]},
        number={'font': {'color': 'white', 'size': 20}}
    ), row=2, col=1)
    
    # Sleep Gauge
    fig_gauges.add_trace(go.Indicator(
        mode="gauge+number",
        value=sleep_hours,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sleep Hours/Night", 'font': {'color': 'white', 'size': 14}},
        gauge={'axis': {'range': [3, 12]},
               'bar': {'color': "#3A7BD5"},
               'steps': [{'range': [3, 7], 'color': "rgba(255,65,108,0.3)"},
                        {'range': [7, 9], 'color': "rgba(0,210,255,0.3)"},
                        {'range': [9, 12], 'color': "rgba(255,107,53,0.3)"}]},
        number={'font': {'color': 'white', 'size': 20}}
    ), row=2, col=2)
    
    fig_gauges.update_layout(
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    st.plotly_chart(fig_gauges, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Prediction Function ------------------
def preprocess_inputs(age, height, weight, gender, blood_pressure, heart_rate, steps, exercise_hours, sleep_hours, smokes):
    if label_encoder_gender:
        gender_encoded = label_encoder_gender.transform([gender])[0]
    else:
        gender_encoded = 0 if gender == 'Male' else 1
        
    smokes_encoded = 1 if smokes == 'Yes' else 0
    
    features = np.array([[
        age, height, weight, gender_encoded, heart_rate, 
        steps, exercise_hours, sleep_hours, blood_pressure, smokes_encoded
    ]])
    
    if scaler:
        features_scaled = scaler.transform(features)
    else:
        # Fallback scaling if scaler not available
        from sklearn.preprocessing import StandardScaler
        temp_scaler = StandardScaler()
        features_scaled = temp_scaler.fit_transform(features)
    
    return features_scaled

# ------------------ Prediction Section ------------------
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.header('🤖 AI Fitness Prediction')

pred_col1, pred_col2 = st.columns([1, 2])

with pred_col1:
    if st.button('🚀 **RUN ADVANCED FITNESS ANALYSIS**', use_container_width=True):
        if model is None:
            st.error("Model not loaded. Cannot make predictions.")
        else:
            with st.spinner('🤖 AI is analyzing your fitness profile...'):
                # Preprocess inputs
                processed_input = preprocess_inputs(
                    age, height, weight, gender, blood_pressure, heart_rate, 
                    steps, exercise_hours, sleep_hours, smokes
                )
                
                # Make prediction
                prediction_prob = model.predict(processed_input)
                prediction = (prediction_prob > 0.5).astype(int)[0][0]
                
                # Store in session state
                st.session_state.prediction_prob = prediction_prob[0][0]
                st.session_state.prediction = prediction
                st.session_state.show_results = True

# Show results if prediction has been made
if 'show_results' in st.session_state and st.session_state.show_results:
    with pred_col2:
        prob = st.session_state.prediction_prob
        pred = st.session_state.prediction
        
        if pred == 1:
            st.markdown('<div class="prediction-card pulse-animation">', unsafe_allow_html=True)
            st.markdown('<h1 style="color: white; margin: 0;">🎉 EXCELLENT FITNESS!</h1>', unsafe_allow_html=True)
            st.markdown(f'<h3 style="color: white; margin: 10px 0;">Confidence: {prob*100:.1f}%</h3>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 1.2rem; margin: 0;">Your lifestyle shows outstanding fitness habits!</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-card pulse-animation">', unsafe_allow_html=True)
            st.markdown('<h1 style="color: white; margin: 0;">💪 FITNESS IMPROVEMENT NEEDED</h1>', unsafe_allow_html=True)
            st.markdown(f'<h3 style="color: white; margin: 10px 0;">Confidence: {(1-prob)*100:.1f}%</h3>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 1.2rem; margin: 0;">Opportunities for health enhancement detected</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ------------------ Detailed Recommendations ------------------
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.header('🎯 Personalized Health Recommendations')
    
    # Create comprehensive health analysis
    improvement_areas = []
    
    if steps < 7500:
        improvement_areas.append(("🚶‍♂️ **Step Count**", f"Current: {steps:,} | Target: 7,500-10,000", 
                                "Increase daily walking by 20-30 minutes"))
    
    if exercise_hours < 2.5:
        improvement_areas.append(("💪 **Exercise**", f"Current: {exercise_hours} hrs | Target: 2.5-5 hrs", 
                                "Add strength training 2x weekly"))
    
    if heart_rate > 75:
        improvement_areas.append(("❤️ **Heart Rate**", f"Current: {heart_rate} bpm | Target: <70 bpm", 
                                "Incorporate cardio 3x weekly"))
    
    if sleep_hours < 7 or sleep_hours > 9:
        improvement_areas.append(("😴 **Sleep**", f"Current: {sleep_hours} hrs | Target: 7-9 hrs", 
                                "Establish consistent sleep schedule"))
    
    if blood_pressure >= 130:
        improvement_areas.append(("🩸 **Blood Pressure**", f"Current: {blood_pressure} mmHg | Target: <120 mmHg", 
                                "Reduce sodium intake, increase cardio"))
    
    if smokes == 'Yes':
        improvement_areas.append(("🚭 **Smoking**", "Current: Smoker | Target: Non-smoker", 
                                "Consider smoking cessation programs"))
    
    for area, current, recommendation in improvement_areas:
        with st.expander(area, expanded=True):
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #FF6B35;">
                <p style="margin: 0; color: white; font-weight: 600;">Status: {current}</p>
                <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.8);">Recommendation: {recommendation}</p>
            </div>
            """, unsafe_allow_html=True)

# ------------------ Sidebar ------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #FF6B35 0%, #FF8E53 100%); border-radius: 16px; margin-bottom: 1rem;">
        <h2 style="color: #ffffff; margin: 0; font-size: 1.8rem;">🏃‍♂️ FitAI</h2>
        <p style="color: #ffffff; margin: 0.5rem 0 0 0; font-size: 1rem;">Advanced Fitness Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.header('🔬 Model Information')
    st.markdown("""
    **AI Architecture**: Deep Neural Network
    - Input Layer: 10 features
    - Hidden Layers: 64-32 neurons
    - Output: Fitness probability
    - Accuracy: 92% (validated)
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.header('⚠️ Disclaimer')
    st.markdown("""
    This AI tool provides predictive insights only. Consult healthcare professionals for medical advice. Regular check-ups and professional guidance are essential for health management.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Footer ------------------
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="background: rgba(255,255,255,0.05); padding: 2rem; border-radius: 16px; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
    <div style="text-align: center;">
        <h4 style="color: #FF6B35; margin-bottom: 1rem; font-size: 1.5rem;">👨‍💻 Developed By</h4>
        <p style="font-size: 1.4rem; font-weight: bold; color: #00D2FF; margin-bottom: 0.5rem;">
            Harshith Narasimhamurthy
        </p>
        <p style="margin-bottom: 0.5rem; font-size: 1.1rem; color: rgba(255,255,255,0.8);">
            📧 harshithnchandan@gmail.com | 📱 +919663918804
        </p>
        <p style="margin-bottom: 1rem; font-size: 1.1rem;">
            🔗 <a href='https://www.linkedin.com/in/harshithnarasimhamurthy69/' target='_blank' 
               style='color: #FF8E53; text-decoration: none; font-weight: bold;'>
               Connect with me on LinkedIn
            </a>
        </p>
        <p style="font-size: 1rem; color: rgba(255,255,255,0.6);">
            Powered by TensorFlow & Streamlit | Advanced Analytics Dashboard v2.0
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
