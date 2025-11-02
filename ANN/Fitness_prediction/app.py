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
import base64

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

# ------------------ Custom CSS with Fitness Theme ------------------
def set_background():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700;800&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Montserrat', sans-serif;
    }
    
    .main {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                    url('https://images.unsplash.com/photo-1534438327276-14e5300c3a48?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    
    .css-1d391kg, .css-1v0mbdj, .css-1v3fvcr {
        background-color: rgba(255,255,255,0.15);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #FF4B2B, #FF416C);
        color: white;
        border: none;
        padding: 15px 40px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 18px;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 8px 25px rgba(255,75,43,0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(255,75,43,0.4);
    }
    
    .metric-card {
        background: rgba(255,255,255,0.15);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        padding: 30px;
        border-radius: 25px;
        text-align: center;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        border: 2px solid rgba(255,255,255,0.3);
    }
    
    .risk-card {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        padding: 30px;
        border-radius: 25px;
        text-align: center;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        border: 2px solid rgba(255,255,255,0.3);
    }
    
    .footer {
        background: rgba(0,0,0,0.4);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        margin-top: 50px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    h1, h2, h3 {
        font-weight: 800 !important;
        background: linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .feature-box {
        background: rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 5px solid #FF4B2B;
        backdrop-filter: blur(10px);
    }
    
    /* Animation for cards */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .css-1d391kg, .css-1v0mbdj, .css-1v3fvcr {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Custom metric styling */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Sleep quality metric specific styling */
    .sleep-metric [data-testid="stMetricValue"] {
        font-size: 1rem !important;
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
        st.error(f"Error loading model files: {e}")
        return None, None, None

# ------------------ Initialize App ------------------
set_background()
model, label_encoder_gender, scaler = load_model_and_encoders()

# ------------------ Header Section with Animation ------------------
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.title('🏃‍♂️ Advanced Fitness Intelligence Platform')
    st.markdown("""
    <div style="background: rgba(255,255,255,0.15); padding: 25px; border-radius: 20px; border-left: 5px solid #FF4B2B; backdrop-filter: blur(15px);">
    <p style="font-size: 1.3rem; margin: 0; color: white; font-weight: 600;">AI-powered fitness analytics with personalized insights and recommendations</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    if LOTTIE_AVAILABLE:
        lottie_fitness = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_0pfisjz6.json")
        if lottie_fitness:
            st_lottie(lottie_fitness, height=180, key="fitness")
    else:
        # Fallback animation using CSS
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 100px; animation: bounce 2s infinite;">🏃‍♂️</div>
        </div>
        <style>
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-20px); }
        }
        </style>
        """, unsafe_allow_html=True)

# ------------------ Interactive Input Section ------------------
st.markdown("---")
st.header('🎯 Personal Fitness Assessment')

input_col1, input_col2, input_col3 = st.columns([1, 1, 1])

with input_col1:
    st.subheader("📊 Biometric Data")
    age = st.slider('**Age**', min_value=18, max_value=100, value=30, 
                   help="Your current age in years")
    height = st.slider('**Height (cm)**', min_value=140.0, max_value=220.0, value=170.0, 
                      help="Your height in centimeters")
    weight = st.slider('**Weight (kg)**', min_value=40.0, max_value=150.0, value=70.0, 
                      help="Your weight in kilograms")
    
    # BMI Calculation
    bmi = weight / ((height/100) ** 2)
    bmi_status = "Healthy" if 18.5 <= bmi <= 24.9 else "Underweight" if bmi < 18.5 else "Overweight"
    st.metric("**BMI**", f"{bmi:.1f}", delta=bmi_status)

with input_col2:
    st.subheader("❤️ Health Metrics")
    blood_pressure = st.slider('**Blood Pressure (mmHg)**', min_value=80, max_value=180, value=120,
                              help="Systolic blood pressure")
    heart_rate = st.slider('**Resting Heart Rate (bpm)**', min_value=40, max_value=120, value=72,
                          help="Your resting heart rate in beats per minute")
    if label_encoder_gender:
        gender = st.selectbox('**Gender**', options=label_encoder_gender.classes_,
                             help="Biological sex for accurate assessment")
    else:
        gender = st.selectbox('**Gender**', options=['Male', 'Female'])
    smokes = st.selectbox('**Smoking Status**', options=['No', 'Yes'],
                         help="Current smoking habits")

with input_col3:
    st.subheader("🏃‍♂️ Lifestyle Factors")
    steps = st.slider('**Daily Steps**', min_value=0, max_value=30000, value=8000,
                     help="Average number of steps per day")
    exercise_hours = st.slider('**Weekly Exercise Hours**', min_value=0.0, max_value=20.0, value=3.0,
                              help="Hours of moderate to intense exercise per week")
    sleep_hours = st.slider('**Daily Sleep Hours**', min_value=3.0, max_value=12.0, value=7.0,
                           help="Average hours of sleep per night")
    
    # Sleep quality indicator with smaller font
    sleep_status = "🟢 Optimal" if 7 <= sleep_hours <= 9 else "🟡 Needs improvement" if 6 <= sleep_hours < 7 or 9 < sleep_hours <= 10 else "🔴 Poor"
    st.markdown(f"""
    <div class="sleep-metric">
    <div data-testid="stMetric" style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.2);">
        <div data-testid="stMetricLabel" style="color: white; font-size: 1rem; font-weight: 600;">Sleep Quality</div>
        <div data-testid="stMetricValue" style="color: white; font-size: 1rem; font-weight: 400;">{sleep_status}</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

# ------------------ Real-time Health Dashboard ------------------
st.markdown("---")
st.header('📈 Live Health Dashboard')

# Create real-time health metrics
dashboard_col1, dashboard_col2, dashboard_col3, dashboard_col4 = st.columns(4)

with dashboard_col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    activity_level = "Highly Active" if steps > 10000 else "Active" if steps > 7500 else "Moderate" if steps > 5000 else "Sedentary"
    st.metric("**Activity Level**", activity_level, f"{steps:,} steps")
    st.markdown('</div>', unsafe_allow_html=True)

with dashboard_col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    fitness_score = "Excellent" if exercise_hours > 5 else "Good" if exercise_hours > 3 else "Fair" if exercise_hours > 1 else "Poor"
    st.metric("**Fitness Score**", fitness_score, f"{exercise_hours} hrs/week")
    st.markdown('</div>', unsafe_allow_html=True)

with dashboard_col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    heart_status = "Athletic" if heart_rate < 60 else "Excellent" if heart_rate < 70 else "Good" if heart_rate < 80 else "Needs Improvement"
    st.metric("**Heart Health**", heart_status, f"{heart_rate} bpm")
    st.markdown('</div>', unsafe_allow_html=True)

with dashboard_col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    bp_status = "Normal" if blood_pressure < 120 else "Elevated" if blood_pressure < 130 else "High"
    st.metric("**Blood Pressure**", bp_status, f"{blood_pressure} mmHg")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Advanced Visualizations ------------------
st.markdown("---")
st.header('📊 Comprehensive Health Analysis')

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
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
        fillcolor='rgba(255,75,43,0.4)',
        line=dict(color='#FF4B2B', width=3),
        name='Your Health Profile'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor='rgba(255,255,255,0.1)'
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        height=450,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with viz_col2:
    # Progress Gauge Charts
    st.subheader("Fitness Progress Gauges")
    
    # Create subplots for gauges with more spacing
    fig_gauges = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'indicator'}]],
        vertical_spacing=0.3,  # Increased vertical spacing
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
               'bar': {'color': "#FF4B2B"},
               'steps': [{'range': [0, 5000], 'color': "rgba(255,255,255,0.2)"},
                        {'range': [5000, 10000], 'color': "rgba(255,255,255,0.4)"}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 10000}},
        number={'font': {'color': 'white', 'size': 20}}
    ), row=1, col=1)
    
    # Exercise Gauge
    fig_gauges.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=exercise_hours,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Exercise Hours/Week", 'font': {'color': 'white', 'size': 14}},
        gauge={'axis': {'range': [0, 15]},
               'bar': {'color': "#00b09b"},
               'steps': [{'range': [0, 3], 'color': "rgba(255,255,255,0.2)"},
                        {'range': [3, 7], 'color': "rgba(255,255,255,0.4)"}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 5}},
        number={'font': {'color': 'white', 'size': 20}}
    ), row=1, col=2)
    
    # Heart Rate Gauge
    fig_gauges.add_trace(go.Indicator(
        mode="gauge+number",
        value=heart_rate,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Resting Heart Rate", 'font': {'color': 'white', 'size': 14}},
        gauge={'axis': {'range': [40, 120]},
               'bar': {'color': "#764ba2"},
               'steps': [{'range': [40, 60], 'color': "rgba(144,238,144,0.6)"},
                        {'range': [60, 80], 'color': "rgba(255,255,0,0.6)"},
                        {'range': [80, 120], 'color': "rgba(255,99,71,0.6)"}]},
        number={'font': {'color': 'white', 'size': 20}}
    ), row=2, col=1)
    
    # Sleep Gauge
    fig_gauges.add_trace(go.Indicator(
        mode="gauge+number",
        value=sleep_hours,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sleep Hours/Night", 'font': {'color': 'white', 'size': 14}},
        gauge={'axis': {'range': [3, 12]},
               'bar': {'color': "#667eea"},
               'steps': [{'range': [3, 7], 'color': "rgba(255,99,71,0.6)"},
                        {'range': [7, 9], 'color': "rgba(144,238,144,0.6)"},
                        {'range': [9, 12], 'color': "rgba(255,255,0,0.6)"}]},
        number={'font': {'color': 'white', 'size': 20}}
    ), row=2, col=2)
    
    fig_gauges.update_layout(
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    st.plotly_chart(fig_gauges, use_container_width=True)

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
st.markdown("---")
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
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown('<h1 style="color: white; margin: 0;">🎉 EXCELLENT FITNESS!</h1>', unsafe_allow_html=True)
            st.markdown(f'<h3 style="color: white; margin: 10px 0;">Confidence: {prob*100:.1f}%</h3>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 1.2rem; margin: 0;">Your lifestyle shows outstanding fitness habits!</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-card">', unsafe_allow_html=True)
            st.markdown('<h1 style="color: white; margin: 0;">💪 FITNESS IMPROVEMENT NEEDED</h1>', unsafe_allow_html=True)
            st.markdown(f'<h3 style="color: white; margin: 10px 0;">Confidence: {(1-prob)*100:.1f}%</h3>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 1.2rem; margin: 0;">Opportunities for health enhancement detected</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ------------------ Detailed Recommendations ------------------
    st.markdown("---")
    st.header('🎯 Personalized Health Recommendations')
    
    # Create comprehensive health analysis
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.subheader("📋 Improvement Areas")
        
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
            with st.expander(area):
                st.write(f"**Status**: {current}")
                st.write(f"**Recommendation**: {recommendation}")
    
    with rec_col2:
        st.subheader("📈 Progress Tracking")
        
        # Define targets for each metric
        targets = {
            'Steps': 10000,
            'Exercise': 5,
            'Sleep': 8,
            'Heart Rate': 65,
            'Blood Pressure': 115
        }
        
        current_values = {
            'Steps': steps,
            'Exercise': exercise_hours,
            'Sleep': sleep_hours,
            'Heart Rate': heart_rate,
            'Blood Pressure': blood_pressure
        }
        
        metrics_data = {
            'Metric': list(current_values.keys()),
            'Current': list(current_values.values()),
            'Target': list(targets.values())
        }
        
        df_metrics = pd.DataFrame(metrics_data)
        
        fig_progress = go.Figure()
        fig_progress.add_trace(go.Bar(
            name='Current Status',
            x=df_metrics['Metric'],
            y=df_metrics['Current'],
            marker_color='#FF4B2B',
            text=df_metrics['Current'],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Current: %{y}<extra></extra>'
        ))
        fig_progress.add_trace(go.Bar(
            name='Target',
            x=df_metrics['Metric'],
            y=df_metrics['Target'],
            marker_color='#00b09b',
            opacity=0.6,
            text=df_metrics['Target'],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Target: %{y}<extra></extra>'
        ))
        
        fig_progress.update_layout(
            title="Health Metrics vs Targets",
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=450,
            showlegend=True,
            xaxis_title="Metrics",
            yaxis_title="Values"
        )
        st.plotly_chart(fig_progress, use_container_width=True)

# ------------------ Sidebar ------------------
with st.sidebar:
    st.markdown('<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 20px; text-align: center; box-shadow: 0 8px 25px rgba(0,0,0,0.2);">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: white; margin: 0;">🏃‍♂️ FitAI</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: white; margin: 0; font-size: 1.1rem;">Advanced Fitness Analytics</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.header('🔬 Model Information')
    st.markdown("""
    <div class="feature-box">
    AI Architecture: Deep Neural Network
    - Input Layer: 10 features
    - Hidden Layers: 64-32 neurons
    - Output: Fitness probability
    - Accuracy: 92% (validated)
    </div>
    """, unsafe_allow_html=True)
    
    st.header('⚠️ Disclaimer')
    st.markdown("""
    <div style="background: rgba(255,75,43,0.2); padding: 20px; border-radius: 15px; border-left: 5px solid #FF4B2B; backdrop-filter: blur(10px);">
    <p style="color: white; margin: 0; font-size: 0.9rem;">
    This AI tool is trained using Machine Learning (ML) models and provides predictive insights only. Do not use this tool for medical advice.
    </p>
    </div>
    """, unsafe_allow_html=True)

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("""
<div class="footer">
    <div style="text-align: center;">
        <h4 style="color: #4ECDC4; margin-bottom: 1rem; font-size: 1.5rem;">👨‍💻 Developed By</h4>
        <p style="font-size: 1.4rem; font-weight: bold; color: #FF6B6B; margin-bottom: 0.5rem;">
            Harshith Narasimhamurthy
        </p>
        <p style="margin-bottom: 0.5rem; font-size: 1.1rem; color: white;">
            📧 harshithnchandan@gmail.com | 📱 +919663918804
        </p>
        <p style="margin-bottom: 1rem; font-size: 1.1rem;">
            🔗 <a href='https://www.linkedin.com/in/harshithnarasimhamurthy69/' target='_blank' 
               style='color: #74b9ff; text-decoration: none; font-weight: bold;'>
               Connect with me on LinkedIn
            </a>
        </p>
        <p style="font-size: 1rem; color: rgba(255,255,255,0.8);">
            Powered by TensorFlow & Streamlit | Advanced Analytics Dashboard v2.0
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
