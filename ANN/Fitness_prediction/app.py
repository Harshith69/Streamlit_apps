import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import pickle
import os

# ------------------ File Paths ------------------
MODEL_DIR = 'ANN/Fitness_prediction'
MODEL_PATH = os.path.join(MODEL_DIR, 'predmodel.h5')
GENDER_ENCODER_PATH = os.path.join(MODEL_DIR, 'gender_encoder.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# ------------------ Check if required files exist ------------------
def check_required_files():
    missing_files = []
    if not os.path.exists(MODEL_PATH):
        missing_files.append(MODEL_PATH)
    if not os.path.exists(GENDER_ENCODER_PATH):
        missing_files.append(GENDER_ENCODER_PATH)
    if not os.path.exists(SCALER_PATH):
        missing_files.append(SCALER_PATH)
    return missing_files

# ------------------ Load model and encoders with error handling ------------------
def load_models():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(GENDER_ENCODER_PATH, 'rb') as file:
            label_encoder_gender = pickle.load(file)
        with open(SCALER_PATH, 'rb') as file:
            scaler = pickle.load(file)
        return model, label_encoder_gender, scaler, None
    except Exception as e:
        return None, None, None, str(e)

# ------------------ Streamlit UI ------------------
st.title('Fitness Level Prediction')
st.write('Predict whether an individual is fit or not based on lifestyle and biometric data')

# Check for missing files first
missing_files = check_required_files()
if missing_files:
    st.error(f"❌ Missing required files: {', '.join(missing_files)}")
    st.info("Please make sure these files are in the same directory as your app.py:")
    st.write("- predmodel.h5 (trained model)")
    st.write("- gender_encoder.pkl (gender encoder)")
    st.write("- scaler.pkl (feature scaler)")
    st.stop()

# Load models
model, label_encoder_gender, scaler, error = load_models()

if error:
    st.error(f"❌ Error loading model files: {error}")
    st.stop()

# Show success message
st.success("✅ All model files loaded successfully!")

# ------------------ User Input Form ------------------
st.header('User Information')
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    height = st.number_input('Height (cm)', min_value=140.0, max_value=220.0, value=170.0)
    weight = st.number_input('Weight (kg)', min_value=40.0, max_value=150.0, value=70.0)
    gender = st.selectbox('Gender', options=label_encoder_gender.classes_)
    blood_pressure = st.number_input('Blood Pressure (mmHg)', min_value=80, max_value=180, value=120)

with col2:
    heart_rate = st.number_input('Resting Heart Rate (bpm)', min_value=40, max_value=120, value=72)
    steps = st.number_input('Daily Steps', min_value=0, max_value=30000, value=8000)
    exercise_hours = st.number_input('Weekly Exercise Hours', min_value=0.0, max_value=20.0, value=3.0)
    sleep_hours = st.number_input('Daily Sleep Hours', min_value=3.0, max_value=12.0, value=7.0)
    smokes = st.selectbox('Smokes', options=['No', 'Yes'])

# ------------------ Preprocessing Function ------------------
def preprocess_inputs(age, height, weight, gender, blood_pressure, heart_rate, steps, exercise_hours, sleep_hours, smokes):
    try:
        # Encode gender
        gender_encoded = label_encoder_gender.transform([gender])[0]
        
        # Encode smokes (convert to 1 for Yes, 0 for No)
        smokes_encoded = 1 if smokes == 'Yes' else 0
        
        # Create feature array in the same order as training data
        features = np.array([[
            age, 
            height, 
            weight, 
            gender_encoded, 
            heart_rate, 
            steps, 
            exercise_hours, 
            sleep_hours,
            blood_pressure,
            smokes_encoded
        ]])
        
        # Scale features using the saved scaler
        features_scaled = scaler.transform(features)
        
        return features_scaled, None
    except Exception as e:
        return None, str(e)

# ------------------ Prediction ------------------
if st.button('Predict Fitness Level'):
    # Preprocess inputs
    processed_input, error = preprocess_inputs(
        age, height, weight, gender, blood_pressure, heart_rate, 
        steps, exercise_hours, sleep_hours, smokes
    )
    
    if error:
        st.error(f"❌ Error preprocessing inputs: {error}")
        st.stop()
    
    # Make prediction
    try:
        prediction_prob = model.predict(processed_input)
        prediction = (prediction_prob > 0.5).astype(int)[0][0]
        
        # Display results
        st.header('Prediction Results')
        
        # Create result columns
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.metric("Prediction Probability", f"{prediction_prob[0][0]:.4f}")
        
        with result_col2:
            if prediction == 1:
                st.success('🏃‍♂️ **FIT** - The model predicts this individual is FIT')
            else:
                st.error('💤 **NOT FIT** - The model predicts this individual is NOT FIT')
        
        # Additional information
        st.subheader('Interpretation')
        st.write('''
        - **FIT**: Individual likely maintains good physical condition through regular exercise and healthy habits
        - **NOT FIT**: Individual may need to improve physical activity levels or lifestyle habits
        ''')
        
        # Feature importance insights
        st.subheader('Health Assessment & Recommendations')
        fitness_tips = []
        
        # Steps analysis
        if steps < 5000:
            fitness_tips.append("🚶 **Steps**: Low step count - aim for 7,000-10,000 daily steps")
        elif steps > 10000:
            fitness_tips.append("✅ **Steps**: Excellent step count!")
        else:
            fitness_tips.append("👍 **Steps**: Good step count - maintain this level")
        
        # Exercise analysis
        if exercise_hours < 2:
            fitness_tips.append("💪 **Exercise**: Try to get at least 2-3 hours of exercise per week")
        elif exercise_hours > 5:
            fitness_tips.append("✅ **Exercise**: Great exercise routine!")
        else:
            fitness_tips.append("👍 **Exercise**: Good exercise frequency")
        
        # Heart rate analysis
        if heart_rate > 80:
            fitness_tips.append("❤️ **Heart Rate**: Could be improved with regular cardio exercise")
        elif heart_rate < 60:
            fitness_tips.append("✅ **Heart Rate**: Excellent resting heart rate (athletic level)")
        else:
            fitness_tips.append("👍 **Heart Rate**: Healthy resting heart rate")
        
        # Sleep analysis
        if sleep_hours < 6:
            fitness_tips.append("😴 **Sleep**: Aim for 7-9 hours of sleep for optimal fitness")
        elif sleep_hours > 9:
            fitness_tips.append("💤 **Sleep**: Good amount of sleep, but excessive sleep may indicate other issues")
        else:
            fitness_tips.append("✅ **Sleep**: Optimal sleep duration")
        
        # Blood pressure analysis
        if blood_pressure < 90:
            fitness_tips.append("📉 **Blood Pressure**: Low blood pressure - monitor and consult if symptomatic")
        elif blood_pressure < 120:
            fitness_tips.append("✅ **Blood Pressure**: Normal blood pressure - excellent!")
        elif blood_pressure < 130:
            fitness_tips.append("📈 **Blood Pressure**: Elevated - consider lifestyle modifications")
        elif blood_pressure < 140:
            fitness_tips.append("⚠️ **Blood Pressure**: Stage 1 hypertension - consult healthcare provider")
        else:
            fitness_tips.append("🚨 **Blood Pressure**: High - seek medical advice")
        
        # Smoking analysis
        if smokes == 'Yes':
            fitness_tips.append("🚭 **Smoking**: Significantly impacts fitness - consider reduction/cessation programs")
        else:
            fitness_tips.append("✅ **Smoking**: Non-smoker - great for overall health!")
        
        # Display all tips
        for tip in fitness_tips:
            st.write(tip)
            
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")

# ------------------ Sidebar ------------------
st.sidebar.header('About This App')
st.sidebar.write('''
This application uses an Artificial Neural Network (ANN) to predict fitness levels based on:

- **Biometric Data**: Age, Height, Weight, Blood Pressure
- **Lifestyle Factors**: Smoking status, Exercise habits, Sleep patterns
- **Physical Metrics**: Resting heart rate, Daily steps
''')

st.sidebar.header('Model Information')
st.sidebar.write('''
- **Model Type**: Artificial Neural Network
- **Architecture**: 3 layers (64-32-1 neurons)
- **Activation**: ReLU (hidden), Sigmoid (output)
- **Training**: Early stopping with validation
''')

st.sidebar.header('Input Guidelines')
st.sidebar.write('''
**Blood Pressure Ranges:**
- Normal: < 120 mmHg
- Elevated: 120-129 mmHg  
- High: ≥ 130 mmHg

**Other Ranges:**
- Age: 18-100 years
- Height: 140-220 cm  
- Weight: 40-150 kg
- Heart Rate: 40-120 bpm
- Steps: 0-30,000 daily
- Exercise: 0-20 hours/week
- Sleep: 3-12 hours daily
''')

# Show current directory info in sidebar for debugging
st.sidebar.header('Debug Info')
st.sidebar.write(f"Current directory: {os.getcwd()}")
st.sidebar.write("Files in directory:")
for file in os.listdir('.'):
    st.sidebar.write(f"- {file}")

# Footer
st.markdown("---")
st.write("*Note: This is a predictive model and should not replace professional medical advice.*")
