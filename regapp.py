import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Salary Predictor Pro",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .feature-card {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .stButton>button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(255,107,107,0.4);
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

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">💰 Salary Predictor Pro</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Salary Estimation Platform")
    
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
        tab1, tab2 = st.tabs(["Personal Info", "Financial Details"])
        
        with tab1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            
            # Personal information inputs
            age = st.slider("Age", 18, 80, 35, help="Customer's age")
            gender = st.selectbox("Gender", ["Female", "Male"])
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            
            # Account information
            tenure = st.slider("Tenure (years)", 0, 10, 5, help="Number of years with the bank")
            num_products = st.slider("Number of Products", 1, 4, 1)
            has_cr_card = st.radio("Has Credit Card", ["Yes", "No"])
            is_active = st.radio("Is Active Member", ["Yes", "No"])
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            
            # Financial inputs
            credit_score = st.slider("Credit Score", 300, 850, 650, 
                                   help="Customer's credit score (300-850)")
            balance = st.number_input("Account Balance ($)", 0.0, 250000.0, 50000.0, 
                                    step=1000.0, format="%.2f")
            exited = st.radio("Previously Exited", ["No", "Yes"])
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 Prediction Results")
        
        # Prediction card
        if st.button("🚀 Predict Salary", use_container_width=True):
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
            
            # Display prediction
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown(f"### 💰 Predicted Salary")
            st.markdown(f"# ${prediction:,.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Additional insights
            st.markdown("### 📈 Salary Insights")
            
            # Create gauge chart for salary range
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prediction,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Salary Range Comparison"},
                delta = {'reference': 100000, 'increasing': {'color': "#4ECDC4"}},
                gauge = {
                    'axis': {'range': [None, 200000]},
                    'bar': {'color': "#FF6B6B"},
                    'steps': [
                        {'range': [0, 50000], 'color': "lightgray"},
                        {'range': [50000, 100000], 'color': "gray"},
                        {'range': [100000, 200000], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 150000
                    }
                }
            ))
            
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Feature importance visualization
            feature_names = [
                'Credit Score', 'Gender', 'Age', 'Tenure', 'Balance',
                'Num Products', 'Has Credit Card', 'Is Active', 'Exited',
                'Geography_France', 'Geography_Germany', 'Geography_Spain'
            ]
            
            # Simple feature impact visualization (this is a simplified version)
            feature_impact = np.abs(features_scaled[0])
            fig_bar = px.bar(
                x=feature_impact,
                y=feature_names,
                orientation='h',
                title="Feature Impact on Prediction",
                color=feature_impact,
                color_continuous_scale='viridis'
            )
            fig_bar.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        else:
            # Placeholder when no prediction is made
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 4rem 2rem; border-radius: 20px; text-align: center; color: white;'>
                <h3>🔮 Ready to Predict</h3>
                <p>Fill in the customer details and click the predict button to get an AI-powered salary estimation!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Sample statistics
            st.markdown("### 📊 Quick Stats")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("Average Credit Score", "650", "15")
            with col_stat2:
                st.metric("Typical Balance", "$75K", "5%")
            with col_stat3:
                st.metric("Common Products", "2", "-1")

    # Footer
        # Footer with personal details
    st.markdown("---")
    
    # Create columns for footer layout
    footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
    
    with footer_col2:
        st.markdown(
            """
            <div style='text-align: center; color: #666; padding: 2rem;'>
                <h4 style='color: #4ECDC4; margin-bottom: 1rem;'>👨‍💻 Developed By</h4>
                <p style='font-size: 1.2rem; font-weight: bold; color: #FF6B6B; margin-bottom: 0.5rem;'>
                    Harshith Narasimhamurthy
                </p>
                <p style='margin-bottom: 0.5rem;'>
                    📧 harshithnchandan@gmail.com | 📱 +919663918804
                </p>
                <p style='margin-bottom: 1rem;'>
                    🔗 <a href='https://www.linkedin.com/in/harshithnarasimhamurthy69/' target='_blank' 
                       style='color: #0077B5; text-decoration: none;'>
                       Connect with me on LinkedIn
                    </a>
                </p>
                <p style='font-size: 0.9rem; color: #888;'>
                    Powered by TensorFlow & Streamlit | Built with ❤️ for Salary Prediction
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )


if __name__ == "__main__":

    main()
