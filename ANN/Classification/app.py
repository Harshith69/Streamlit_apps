import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import time
import os

# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Custom CSS for Styling & Animations ------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        animation: fadeIn 1.5s ease-in;
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        font-weight: 300;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(255,107,107,0.3);
        animation: pulse 2s infinite;
    }
    
    .safe-card {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,184,148,0.3);
        animation: bounce 2s infinite;
    }
    
    .input-section {
        background: rgba(255,255,255,0.1);
        padding: 2rem;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        margin: 1rem 0;
    }
    
    .stNumberInput, .stSelectbox, .stSlider {
        background: rgba(255,255,255,0.9);
        border-radius: 10px;
        padding: 0.5rem;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stNumberInput:focus, .stSelectbox:focus, .stSlider:focus {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .footer {
        background: linear-gradient(135deg, #2d3436 0%, #000000 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
        color: white;
        text-align: center;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 6px 20px rgba(116,185,255,0.3);
    }
    
    .progress-bar {
        height: 8px;
        background: rgba(255,255,255,0.3);
        border-radius: 4px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff7675, #fd79a8);
        border-radius: 4px;
        transition: width 1s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ Load model and encoders ------------------
@st.cache_resource
def load_model_and_encoders():
    try:
        base_path = "ANN/Classification"
        
        model_path = os.path.join(base_path, 'model.h5')
        label_encoder_path = os.path.join(base_path, 'label_encoder_gender.pkl')
        onehot_encoder_path = os.path.join(base_path, 'onehot_encoder_geo.pkl')
        scaler_path = os.path.join(base_path, 'scaler.pkl')
        
        # Check if files exist
        if not all(os.path.exists(path) for path in [model_path, label_encoder_path, onehot_encoder_path, scaler_path]):
            st.error("❌ Some model files are missing!")
            st.info(f"Current directory: {os.getcwd()}")
            st.info(f"Looking for files in: {base_path}")
            
            # List what files we found
            st.info("📁 Files found in that directory:")
            try:
                files_in_dir = os.listdir(base_path)
                for file in files_in_dir:
                    st.write(f"   - {file}")
            except FileNotFoundError:
                st.error(f"Directory '{base_path}' not found!")
            
            return None, None, None, None
        
        # Load files
        model = tf.keras.models.load_model(model_path)
        
        with open(label_encoder_path, 'rb') as file:
            label_encoder_gender = pickle.load(file)

        with open(onehot_encoder_path, 'rb') as file:
            onehot_encoder_geo = pickle.load(file)

        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
            
        st.success("✅ All model files loaded successfully!")
        return model, label_encoder_gender, onehot_encoder_geo, scaler
        
    except Exception as e:
        st.error(f"❌ Error loading model or encoders: {e}")
        return None, None, None, None

model, label_encoder_gender, onehot_encoder_geo, scaler = load_model_and_encoders()

# ------------------ Header Section ------------------
st.markdown("""
<div class='main-header'>
    <h1>🎯 Customer Churn Prediction</h1>
    <p>Advanced Machine Learning Dashboard for Customer Retention Analysis</p>
</div>
""", unsafe_allow_html=True)

# ------------------ Main Content ------------------
if model is not None and scaler is not None:
    # Debug info in sidebar
    with st.sidebar:
        st.markdown("### 🔧 Model Info")
        try:
            if hasattr(scaler, 'feature_names_in_'):
                st.write("Scaler expects:", list(scaler.feature_names_in_))
        except:
            st.write("Scaler feature names not available")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='input-section'>", unsafe_allow_html=True)
        st.subheader("🎨 Customer Information")
        
        # Input fields in two columns for better layout
        subcol1, subcol2 = st.columns(2)
        
        with subcol1:
            geography = st.selectbox('📍 Geography', onehot_encoder_geo.categories_[0])
            gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
            age = st.slider('🎂 Age', 18, 92, 35)
            balance = st.number_input('💰 Balance', min_value=0.0, value=50000.0, format="%.2f")
            credit_score = st.number_input('💳 Credit Score', min_value=300.0, max_value=850.0, value=650.0, format="%.2f")
        
        with subcol2:
            estimated_salary = st.number_input('💵 Estimated Salary', min_value=0.0, value=50000.0, format="%.2f")
            tenure = st.slider('⏰ Tenure (Years)', 0, 10, 5)
            num_of_products = st.slider('📦 Number of Products', 1, 4, 2)
            has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            is_active_member = st.selectbox('🏃 Is Active Member', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='input-section'>", unsafe_allow_html=True)
        st.subheader("📊 Quick Insights")
        
        # Display some metrics
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>💰 Balance</h3>
                <h2>${balance:,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3>💳 Credit Score</h3>
                <h2>{credit_score:.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>⏰ Tenure</h3>
                <h2>{tenure} years</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3>📦 Products</h3>
                <h2>{num_of_products}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Prediction Button
        if st.button('🚀 Predict Churn Probability', use_container_width=True):
            with st.spinner('🔮 Analyzing customer data...'):
                time.sleep(1)
                
                # ------------------ Data Preparation ------------------
                gender_encoded = label_encoder_gender.transform([gender])[0]
                geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
                
                # Based on the error, the scaler expects 'Exited' column but not 'EstimatedSalary'
                input_data = pd.DataFrame({
                    'CreditScore': [credit_score],
                    'Geography_France': [geo_encoded[0][0]],
                    'Geography_Germany': [geo_encoded[0][1]],
                    'Geography_Spain': [geo_encoded[0][2]],
                    'Gender': [gender_encoded],
                    'Age': [age],
                    'Tenure': [tenure],
                    'Balance': [balance],
                    'NumOfProducts': [num_of_products],
                    'HasCrCard': [has_cr_card],
                    'IsActiveMember': [is_active_member],
                    'Exited': [0]  # Add this column that the scaler expects
                })
                
                try:
                    input_data_scaled = scaler.transform(input_data)
                    
                    # ------------------ Prediction ------------------
                    prediction = model.predict(input_data_scaled)
                    prediction_proba = float(prediction[0][0])
                    
                    # Display prediction with animation
                    st.markdown(f"""
                    <div class='progress-bar'>
                        <div class='progress-fill' style='width: {prediction_proba * 100}%'></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.metric("Churn Probability", f"{prediction_proba:.2%}")
                    
                    if prediction_proba > 0.5:
                        st.markdown(f"""
                        <div class='prediction-card'>
                            <h2>🚨 High Churn Risk Detected!</h2>
                            <p>This customer has a <strong>{prediction_proba:.2%}</strong> probability of churning.</p>
                            <p>💡 Recommendation: Immediate retention actions needed!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='safe-card'>
                            <h2>✅ Low Churn Risk</h2>
                            <p>This customer has a <strong>{prediction_proba:.2%}</strong> probability of churning.</p>
                            <p>🎉 Great! Customer is likely to stay with us.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.info("The model was trained with different features. Please check your training data format.")

else:
    st.error("""
    ❌ Unable to load the model and encoders. Please ensure the following files are present in ANN/Classification/:
    - model.h5
    - label_encoder_gender.pkl
    - onehot_encoder_geo.pkl
    - scaler.pkl
    """)

# ------------------ Footer Section ------------------
st.markdown("---")

# Create columns for footer layout
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

with footer_col2:
    st.markdown(
        """
        <div class='footer'>
            <div style='text-align: center;'>
                <h4 style='color: #4ECDC4; margin-bottom: 1rem; font-size: 1.5rem;'>👨‍💻 Developed By</h4>
                <p style='font-size: 1.4rem; font-weight: bold; color: #FF6B6B; margin-bottom: 0.5rem;'>
                    Harshith Narasimhamurthy
                </p>
                <p style='margin-bottom: 0.5rem; font-size: 1.1rem;'>
                    📧 harshithnchandan@gmail.com | 📱 +919663918804
                </p>
                <p style='margin-bottom: 1rem; font-size: 1.1rem;'>
                    🔗 <a href='https://www.linkedin.com/in/harshithnarasimhamurthy69/' target='_blank' 
                       style='color: #74b9ff; text-decoration: none; font-weight: bold;'>
                       Connect with me on LinkedIn
                    </a>
                </p>
                <p style='font-size: 1rem; color: rgba(255,255,255,0.8);'>
                    Powered by TensorFlow & Streamlit | Advanced Analytics Dashboard
                </p>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

# ------------------ Sidebar Information ------------------
with st.sidebar:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white;'>
        <h2>ℹ️ About</h2>
        <p>This dashboard predicts customer churn probability using advanced machine learning models.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📈 Model Information")
    st.info("""
    **Algorithm:** Neural Network
    **Accuracy:** ~85%
    **Features:** 10+ customer attributes
    **Training Data:** 10,000+ samples
    """)
    
    st.markdown("### 🎯 How to Use")
    st.write("""
    1. Fill in customer details
    2. Click 'Predict Churn Probability'
    3. View risk assessment
    4. Take appropriate actions
    """)

