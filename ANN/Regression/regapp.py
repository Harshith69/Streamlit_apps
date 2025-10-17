import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# Page configuration
st.set_page_config(
    page_title="Salary Predictor Pro",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with more gradients and animations
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-attachment: fixed;
    }
    
    /* Animated header */
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
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 25%, #45B7D1 50%, #96CEB4 75%, #FFEAA7 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 3s ease infinite, fadeInUp 1s ease-out;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
        padding: 1rem;
    }
    
    .subtitle {
        font-size: 1.5rem;
        text-align: center;
        color: white;
        margin-bottom: 3rem;
        animation: fadeInUp 1s ease-out 0.3s both;
        font-weight: 300;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 25px;
        color: white;
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
        animation: fadeInUp 0.8s ease-out, float 4s ease-in-out infinite;
        border: 1px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(10px);
    }
    
    .feature-card {
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(15px);
        padding: 1.8rem;
        border-radius: 20px;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.3);
        animation: fadeInUp 0.6s ease-out;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        background: rgba(255,255,255,0.2);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 50%, #FF6B6B 100%);
        background-size: 200% 200%;
        color: white;
        border: none;
        padding: 1rem 2.5rem;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 600;
        transition: all 0.4s ease;
        animation: pulse 2s infinite;
        box-shadow: 0 8px 25px rgba(255,107,107,0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(255,107,107,0.6);
        background-position: right center;
        animation: none;
    }
    
    /* Animated stats cards */
    .stat-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        animation: fadeInUp 0.8s ease-out;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(116,185,255,0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background: rgba(255,255,255,0.1);
        border-radius: 10px 10px 0px 0px;
        gap: 1rem;
        padding: 1rem 2rem;
        color: white;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%) !important;
        color: white !important;
    }
    
    /* Custom metric styling */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        padding: 1rem;
        border-radius: 15px;
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(135deg, rgba(0,0,0,0.3) 0%, rgba(0,0,0,0.5) 100%);
        backdrop-filter: blur(15px);
        padding: 2rem;
        border-radius: 20px;
        margin-top: 3rem;
        border: 1px solid rgba(255,255,255,0.1);
        animation: fadeInUp 1s ease-out;
    }
    
    /* Loading animation */
    .loading-dots {
        display: inline-block;
        position: relative;
        width: 80px;
        height: 80px;
    }
    
    .loading-dots div {
        position: absolute;
        top: 33px;
        width: 13px;
        height: 13px;
        border-radius: 50%;
        background: #fff;
        animation-timing-function: cubic-bezier(0, 1, 1, 0);
    }
    
    .loading-dots div:nth-child(1) {
        left: 8px;
        animation: loading-dots1 0.6s infinite;
    }
    
    .loading-dots div:nth-child(2) {
        left: 8px;
        animation: loading-dots2 0.6s infinite;
    }
    
    .loading-dots div:nth-child(3) {
        left: 32px;
        animation: loading-dots2 0.6s infinite;
    }
    
    .loading-dots div:nth-child(4) {
        left: 56px;
        animation: loading-dots3 0.6s infinite;
    }
    
    @keyframes loading-dots1 {
        0% { transform: scale(0); }
        100% { transform: scale(1); }
    }
    
    @keyframes loading-dots3 {
        0% { transform: scale(1); }
        100% { transform: scale(0); }
    }
    
    @keyframes loading-dots2 {
        0% { transform: translate(0, 0); }
        100% { transform: translate(24px, 0); }
    }
</style>
""", unsafe_allow_html=True)

# Load model and preprocessing objects
@st.cache_resource
def load_artifacts():
    model = load_model('regression_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder_gender.pkl', 'rb') as f:
        label_encoder_gender = pickle.load(f)
    with open('onehot_encoder_geo.pkl', 'rb') as f:
        onehot_encoder_geo = pickle.load(f)
    return model, scaler, label_encoder_gender, onehot_encoder_geo

# Loading animation component
def loading_animation():
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; padding: 2rem;">
        <div class="loading-dots">
            <div></div>
            <div></div>
            <div></div>
            <div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main app
def main():
    # Header with enhanced styling
    st.markdown('<h1 class="main-header">💰 Salary Predictor Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Salary Estimation Platform with Advanced Analytics</p>', unsafe_allow_html=True)
    
    try:
        model, scaler, label_encoder_gender, onehot_encoder_geo = load_artifacts()
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return

    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Customer Profile")
        
        # Create tabs for different input sections
        tab1, tab2 = st.tabs(["🎯 Personal Info", "💳 Financial Details"])
        
        with tab1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            
            # Personal information inputs
            age = st.slider("**Age**", 18, 80, 35, help="Customer's age")
            gender = st.selectbox("**Gender**", ["Female", "Male"])
            geography = st.selectbox("**Geography**", ["France", "Germany", "Spain"])
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            
            # Account information
            tenure = st.slider("**Tenure (years)**", 0, 10, 5, help="Number of years with the bank")
            num_products = st.slider("**Number of Products**", 1, 4, 1)
            has_cr_card = st.radio("**Has Credit Card**", ["Yes", "No"])
            is_active = st.radio("**Is Active Member**", ["Yes", "No"])
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            
            # Financial inputs
            credit_score = st.slider("**Credit Score**", 300, 850, 650, 
                                   help="Customer's credit score (300-850)")
            balance = st.number_input("**Account Balance ($)**", 0.0, 250000.0, 50000.0, 
                                    step=1000.0, format="%.2f")
            exited = st.radio("**Previously Exited**", ["No", "Yes"])
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 Prediction Results")
        
        # Prediction card
        if st.button("🚀 Predict Salary", use_container_width=True, key="predict_btn"):
            # Show loading animation
            with st.spinner(''):
                loading_placeholder = st.empty()
                loading_placeholder.markdown("""
                <div style="text-align: center; padding: 2rem;">
                    <div class="loading-dots">
                        <div></div>
                        <div></div>
                        <div></div>
                        <div></div>
                    </div>
                    <p style="color: white; margin-top: 1rem;">Analyzing customer data...</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Simulate processing time for animation
                time.sleep(1.5)
                
                # Prepare input data
                gender_encoded = 0 if gender == "Female" else 1
                has_cr_card_encoded = 1 if has_cr_card == "Yes" else 0
                is_active_encoded = 1 if is_active == "Yes" else 0
                exited_encoded = 1 if exited == "Yes" else 0
                
                # One-hot encode geography
                geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()[0]
                
                # Create feature array
                features = np.array([[
                    credit_score, gender_encoded, age, tenure, balance, 
                    num_products, has_cr_card_encoded, is_active_encoded, 
                    exited_encoded
                ]])
                
                # Add geography encoded features
                features = np.concatenate([features, geo_encoded.reshape(1, -1)], axis=1)
                
                # Scale features
                features_scaled = scaler.transform(features)
                
                # Make prediction
                prediction = model.predict(features_scaled)[0][0]
                
                # Clear loading animation
                loading_placeholder.empty()
            
            # Display prediction with enhanced animation
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown(f"### 💰 Predicted Annual Salary")
            st.markdown(f"<h1 style='text-align: center; font-size: 3.5rem; margin: 1rem 0;'>${prediction:,.2f}</h1>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Additional insights
            st.markdown("### 📈 Salary Insights")
            
            # Create gauge chart for salary range with enhanced styling
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prediction,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Salary Range Comparison", 'font': {'color': 'white', 'size': 20}},
                delta = {'reference': 100000, 'increasing': {'color': "#4ECDC4"}, 'font': {'color': 'white'}},
                gauge = {
                    'axis': {'range': [None, 200000], 'tickcolor': "white", 'tickfont': {'color': 'white'}},
                    'bar': {'color': "#FF6B6B"},
                    'bgcolor': "rgba(0,0,0,0.3)",
                    'borderwidth': 2,
                    'bordercolor': "white",
                    'steps': [
                        {'range': [0, 50000], 'color': "rgba(255,255,255,0.2)"},
                        {'range': [50000, 100000], 'color': "rgba(255,255,255,0.4)"},
                        {'range': [100000, 200000], 'color': "rgba(255,255,255,0.6)"}
                    ],
                    'threshold': {
                        'line': {'color': "#FF6B6B", 'width': 4},
                        'thickness': 0.75,
                        'value': 150000
                    }
                }
            ))
            
            fig_gauge.update_layout(
                height=300, 
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "white"}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Feature importance visualization with enhanced styling
            feature_names = [
                'Credit Score', 'Gender', 'Age', 'Tenure', 'Balance',
                'Num Products', 'Has Credit Card', 'Is Active', 'Exited',
                'Geography_France', 'Geography_Germany', 'Geography_Spain'
            ]
            
            # Simple feature impact visualization
            feature_impact = np.abs(features_scaled[0])
            fig_bar = px.bar(
                x=feature_impact,
                y=feature_names,
                orientation='h',
                title="Feature Impact on Prediction",
                color=feature_impact,
                color_continuous_scale='viridis'
            )
            fig_bar.update_layout(
                showlegend=False, 
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "white"},
                title_font_color="white"
            )
            fig_bar.update_xaxes(color='white')
            fig_bar.update_yaxes(color='white')
            st.plotly_chart(fig_bar, use_container_width=True)
        
        else:
            # Enhanced placeholder when no prediction is made
            st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(102,126,234,0.8) 0%, rgba(118,75,162,0.8) 100%); 
                        padding: 4rem 2rem; border-radius: 25px; text-align: center; color: white;
                        backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.3);
                        animation: float 6s ease-in-out infinite;'>
                <h3 style='font-size: 2rem; margin-bottom: 1rem;'>🔮 Ready to Predict</h3>
                <p style='font-size: 1.2rem; opacity: 0.9;'>Fill in the customer details and click the predict button to get an AI-powered salary estimation!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced sample statistics with animated cards
            st.markdown("### 📊 Quick Stats")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.metric("Average Credit Score", "650", "15")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_stat2:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.metric("Typical Balance", "$75K", "5%")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_stat3:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.metric("Common Products", "2", "-1")
                st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced Footer
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


if __name__ == "__main__":

    main()

