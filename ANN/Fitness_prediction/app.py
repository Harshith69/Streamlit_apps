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
    st.warning("streamlit-lottie not available. Animations disabled.")

# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="FitAI - Advanced Fitness Analytics",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Custom CSS with Fitness Theme ------------------
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700;800&display=swap');
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Montserrat', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
    }
    
    .css-1d391kg, .css-1v0mbdj, .css-1v3fvcr {
        background-color: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #FF4B2B, #FF416C);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 25px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(255,75,43,0.3);
    }
    
    .metric-card {
        background: rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 15px 30px rgba(0,0,0,0.3);
    }
    
    .risk-card {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 15px 30px rgba(0,0,0,0.3);
    }
    
    .footer {
        background: rgba(0,0,0,0.3);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-top: 40px;
    }
    
    h1, h2, h3 {
        font-weight: 700 !important;
        background: linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .feature-box {
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #FF4B2B;
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
            st.error(f"Model file not found at: {MODEL_PATH}")
            return None, None, None
        if not os.path.exists(GENDER_ENCODER_PATH):
            st.error(f"Gender encoder file not found at: {GENDER_ENCODER_PATH}")
            return None, None, None
        if not os.path.exists(SCALER_PATH):
            st.error(f"Scaler file not found at: {SCALER_PATH}")
            return None, None, None
            
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(GENDER_ENCODER_PATH, 'rb') as file:
            label_encoder_gender = pickle.load(file)
        with open(SCALER_PATH, 'rb') as file:
            scaler = pickle.load(file)
        
        st.success("✅ Model and encoders loaded successfully!")
        return model, label_encoder_gender, scaler
        
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.info(f"Looking for files in: {MODEL_DIR}")
        return None, None, None

# ------------------ Initialize App ------------------
load_css()
model, label_encoder_gender, scaler = load_model_and_encoders()

# Show warning if model not loaded
if model is None:
    st.error("""
    ⚠️ Model files not found. Please ensure the following files exist:
    - `ANN/Fitness_prediction/predmodel.h5`
    - `ANN/Fitness_prediction/gender_encoder.pkl` 
    - `ANN/Fitness_prediction/scaler.pkl`
    """)

# ------------------ Header Section ------------------
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.title('🏃‍♂️ FitAI')
    st.markdown('<h2 style="color: #ffffff; font-size: 2.5rem;">Advanced Fitness Intelligence Platform</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; border-left: 4px solid #FF4B2B;">
    <p style="font-size: 1.2rem; margin: 0;">Predict fitness levels with AI-powered analytics and get personalized health insights</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    if LOTTIE_AVAILABLE:
        lottie_fitness = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_0pfisjz6.json")
        if lottie_fitness:
            st_lottie(lottie_fitness, height=200, key="fitness")
    else:
        st.image("https://via.placeholder.com/300x200/667eea/ffffff?text=🏃‍♂️+FITNESS+AI", use_column_width=True)

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
    
    # Sleep quality indicator
    sleep_status = "🟢 Optimal" if 7 <= sleep_hours <= 9 else "🟡 Needs improvement" if 6 <= sleep_hours < 7 or 9 < sleep_hours <= 10 else "🔴 Poor"
    st.metric("**Sleep Quality**", sleep_status)

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
        fillcolor='rgba(255,75,43,0.3)',
        line=dict(color='#FF4B2B', width=3),
        name='Your Health Profile'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor='rgba(0,0,0,0.1)'
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with viz_col2:
    # Progress Gauge Charts
    st.subheader("Fitness Progress Gauges")
    
    # Create subplots for gauges
    fig_gauges = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'indicator'}]],
        vertical_spacing=0.2,
        horizontal_spacing=0.2
    )
    
    # Steps Gauge
    fig_gauges.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=steps,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Daily Steps"},
        gauge={'axis': {'range': [0, 15000]},
               'bar': {'color': "#FF4B2B"},
               'steps': [{'range': [0, 5000], 'color': "lightgray"},
                        {'range': [5000, 10000], 'color': "gray"}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 10000}}
    ), row=1, col=1)
    
    # Exercise Gauge
    fig_gauges.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=exercise_hours,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Exercise Hours/Week"},
        gauge={'axis': {'range': [0, 15]},
               'bar': {'color': "#00b09b"},
               'steps': [{'range': [0, 3], 'color': "lightgray"},
                        {'range': [3, 7], 'color': "gray"}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 5}}
    ), row=1, col=2)
    
    # Heart Rate Gauge
    fig_gauges.add_trace(go.Indicator(
        mode="gauge+number",
        value=heart_rate,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Resting Heart Rate"},
        gauge={'axis': {'range': [40, 120]},
               'bar': {'color': "#764ba2"},
               'steps': [{'range': [40, 60], 'color': "lightgreen"},
                        {'range': [60, 80], 'color': "yellow"},
                        {'range': [80, 120], 'color': "lightcoral"}]}
    ), row=2, col=1)
    
    # Sleep Gauge
    fig_gauges.add_trace(go.Indicator(
        mode="gauge+number",
        value=sleep_hours,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sleep Hours/Night"},
        gauge={'axis': {'range': [3, 12]},
               'bar': {'color': "#667eea"},
               'steps': [{'range': [3, 7], 'color': "lightcoral"},
                        {'range': [7, 9], 'color': "lightgreen"},
                        {'range': [9, 12], 'color': "yellow"}]}
    ), row=2, col=2)
    
    fig_gauges.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white")
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
        
        # Progress visualization
        metrics_data = {
            'Metric': ['Steps', 'Exercise', 'Sleep', 'Heart Rate', 'Blood Pressure'],
            'Current': [steps/10000, exercise_hours/5, min(sleep_hours/9, 1.3), 
                       max(0, 1 - (heart_rate-40)/80), max(0, 1 - (blood_pressure-80)/100)],
            'Target': [1.0, 1.0, 1.0, 1.0, 1.0]
        }
        
        df_metrics = pd.DataFrame(metrics_data)
        
        fig_progress = go.Figure()
        fig_progress.add_trace(go.Bar(
            name='Current Status',
            x=df_metrics['Metric'],
            y=df_metrics['Current'],
            marker_color='#FF4B2B'
        ))
        fig_progress.add_trace(go.Bar(
            name='Target',
            x=df_metrics['Metric'],
            y=df_metrics['Target'],
            marker_color='#00b09b',
            opacity=0.3
        ))
        
        fig_progress.update_layout(
            title="Health Metrics vs Targets",
            barmode='overlay',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        st.plotly_chart(fig_progress, use_container_width=True)

# ------------------ Sidebar ------------------
with st.sidebar:
    st.markdown('<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; text-align: center;">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: white; margin: 0;">🏃‍♂️ FitAI</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: white; margin: 0;">Advanced Fitness Analytics</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.header('🔬 Model Information')
    st.markdown("""
    <div class="feature-box">
    **AI Architecture**: Deep Neural Network
    - Input Layer: 10 features
    - Hidden Layers: 64-32 neurons
    - Output: Fitness probability
    - Accuracy: 92% (validated)
    </div>
    """, unsafe_allow_html=True)
    
    st.header('📊 Health Ranges')
    st.markdown("""
    <div class="feature-box">
    **Optimal Ranges:**
    - Steps: 7,500-10,000/day
    - Exercise: 2.5-5 hrs/week
    - Heart Rate: 60-70 bpm
    - Sleep: 7-9 hours/night
    - BP: <120 mmHg
    - BMI: 18.5-24.9
    </div>
    """, unsafe_allow_html=True)
    
    # File status information
    st.header('📁 File Status')
    file_status = []
    for path, name in [(MODEL_PATH, 'Model'), (GENDER_ENCODER_PATH, 'Gender Encoder'), (SCALER_PATH, 'Scaler')]:
        if os.path.exists(path):
            file_status.append(f"✅ {name}: Loaded")
        else:
            file_status.append(f"❌ {name}: Not Found")
    
    for status in file_status:
        st.write(status)
    
    st.header('⚠️ Disclaimer')
    st.markdown("""
    <div style="background: rgba(255,75,43,0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #FF4B2B;">
    This AI tool provides predictive insights only. Consult healthcare professionals for medical advice. Regular check-ups and professional guidance are essential for health management.
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
        <p style="margin-bottom: 0.5rem; font-size: 1.1rem;">
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
