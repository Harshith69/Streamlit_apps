import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import time
import random

# Page configuration
st.set_page_config(
    page_title="Netflix Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Netflix theme with animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Netflix+Sans:wght@300;400;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #141414 0%, #000000 100%);
        color: #ffffff;
        font-family: 'Netflix Sans', sans-serif;
    }
    
    .netflix-header {
        background: linear-gradient(90deg, #E50914 0%, #B81D24 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(229, 9, 20, 0.3);
        animation: headerGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes headerGlow {
        from { box-shadow: 0 8px 32px rgba(229, 9, 20, 0.3); }
        to { box-shadow: 0 8px 48px rgba(229, 9, 20, 0.6); }
    }
    
    .netflix-title {
        font-family: 'Bebas Neue', cursive;
        font-size: 4rem;
        color: #ffffff;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.5);
        margin: 0;
        letter-spacing: 3px;
    }
    
    .netflix-subtitle {
        font-size: 1.5rem;
        color: #E6E6E6;
        margin-top: 0.5rem;
    }
    
    .review-box {
        background: rgba(45, 45, 45, 0.9);
        border: 2px solid #E50914;
        border-radius: 10px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 4px 20px rgba(229, 9, 20, 0.2);
        transition: all 0.3s ease;
    }
    
    .review-box:hover {
        box-shadow: 0 8px 30px rgba(229, 9, 20, 0.4);
        transform: translateY(-5px);
    }
    
    .sentiment-positive {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        animation: positivePulse 2s infinite;
        box-shadow: 0 4px 20px rgba(0, 176, 155, 0.4);
    }
    
    @keyframes positivePulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .sentiment-negative {
        background: linear-gradient(135deg, #E50914 0%, #FF416C 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        animation: negativeShake 0.5s ease-in-out;
        box-shadow: 0 4px 20px rgba(229, 9, 20, 0.4);
    }
    
    @keyframes negativeShake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
    
    .prediction-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    
    .netflix-button {
        background: linear-gradient(90deg, #E50914 0%, #B81D24 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 5px;
        font-size: 1.2rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    
    .netflix-button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(229, 9, 20, 0.4);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #141414 0%, #000000 100%);
    }
    
    .footer {
        background: rgba(20, 20, 20, 0.9);
        padding: 2rem;
        border-radius: 10px;
        margin-top: 3rem;
        border-top: 3px solid #E50914;
    }
    
    .loading-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
    }
    
    .netflix-loader {
        width: 50px;
        height: 50px;
        border: 5px solid #333;
        border-top: 5px solid #E50914;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .movie-reel {
        display: flex;
        justify-content: center;
        margin: 1rem 0;
        animation: scrollReel 20s linear infinite;
    }
    
    @keyframes scrollReel {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    
    .confidence-bar {
        height: 20px;
        background: #333;
        border-radius: 10px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem;'>
        <h2 style='color: #E50914; border-bottom: 2px solid #E50914; padding-bottom: 0.5rem;'>🎯 Dashboard Info</h2>
        
        <h3 style='color: #FFFFFF; margin-top: 1.5rem;'>📊 About This App</h3>
        <p style='color: #E6E6E6;'>This Netflix-themed sentiment analysis tool uses a trained RNN model to classify movie reviews as positive or negative with cinematic animations.</p>
        
        <h3 style='color: #FFFFFF; margin-top: 1.5rem;'>🎬 How to Use</h3>
        <ol style='color: #E6E6E6; padding-left: 1.2rem;'>
            <li>Type or paste a movie review in the text area</li>
            <li>Click the "Analyze Sentiment" button</li>
            <li>Watch the animated results with confidence score</li>
            <li>Try different reviews to see various animations</li>
        </ol>
        
        <h3 style='color: #FFFFFF; margin-top: 1.5rem;'>🤖 Model Info</h3>
        <ul style='color: #E6E6E6; padding-left: 1.2rem;'>
            <li><strong>Architecture:</strong> Simple RNN</li>
            <li><strong>Training Data:</strong> IMDB Movie Reviews</li>
            <li><strong>Accuracy:</strong> ~85% on test data</li>
            <li><strong>Vocabulary:</strong> 10,000 most frequent words</li>
        </ul>
        
        <div style='background: rgba(229, 9, 20, 0.1); padding: 1rem; border-radius: 5px; margin-top: 2rem;'>
            <h4 style='color: #E50914;'>💡 Pro Tip</h4>
            <p style='color: #E6E6E6; margin: 0;'>For best results, write reviews similar to IMDB format - focus on plot, acting, direction, and overall enjoyment.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main content
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div class='netflix-header'>
        <h1 class='netflix-title'>NETFLIX SENTIMENT</h1>
        <p class='netflix-subtitle'>AI-Powered Movie Review Analysis</p>
    </div>
    """, unsafe_allow_html=True)

# Movie reel animation
st.markdown("<div class='movie-reel'>🎬 🎭 🎪 🎫 📽️ 🎞️ 🍿 🎦</div>", unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("<div class='review-box'>", unsafe_allow_html=True)
    st.markdown("### 🎭 Write Your Movie Review")
    st.markdown("Share your thoughts about any movie and let AI analyze the sentiment!")
    
    user_input = st.text_area(
        "",
        height=200,
        placeholder="Type your movie review here...\n\nExample: 'This movie was absolutely fantastic! The acting was superb, the storyline was engaging, and the cinematography was breathtaking. Highly recommended!'",
        label_visibility="collapsed"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='review-box'>", unsafe_allow_html=True)
    st.markdown("### 📊 Quick Analysis Tips")
    st.markdown("""
    - **Positive words**: amazing, fantastic, brilliant, excellent, wonderful
    - **Negative words**: terrible, awful, boring, disappointing, waste
    - **Neutral context matters**: 'not bad' vs 'bad'
    - **Length**: Longer reviews often provide more context
    """)
    
    # Sample reviews
    sample_reviews = [
        "This movie was absolutely incredible! The acting was phenomenal and the plot kept me on the edge of my seat.",
        "Terrible film. Poor acting, weak storyline, and awful direction. Complete waste of time.",
        "The cinematography was beautiful but the characters were poorly developed and the pacing was too slow."
    ]
    
    if st.button("🎲 Try Sample Review"):
        sample_review = random.choice(sample_reviews)
        st.session_state.sample_review = sample_review
        st.rerun()
    
    if 'sample_review' in st.session_state:
        user_input = st.session_state.sample_review
        st.text_area("", value=user_input, height=100, label_visibility="collapsed")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Analyze button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_clicked = st.button(
        "🎬 ANALYZE SENTIMENT", 
        use_container_width=True, 
        type="primary"
    )

# Load model and process (you'll need to implement this part)
def load_sentiment_model():
    """Placeholder for model loading - replace with your actual model"""
    try:
        # This is where you'd load your actual model
        # model = load_model('simple_rnn_imdb.h5')
        # word_index = imdb.get_word_index()
        # return model, word_index
        return None, None
    except:
        return None, None

def preprocess_text(text, word_index):
    """Placeholder for text preprocessing"""
    # Implement your preprocessing logic here
    return np.random.random()

def predict_sentiment(text, model, word_index):
    """Placeholder for prediction - returns random results for demo"""
    # Replace with actual prediction
    score = np.random.uniform(0, 1)
    sentiment = "Positive" if score > 0.5 else "Negative"
    return sentiment, score

# Process when analyze is clicked
if analyze_clicked and user_input.strip():
    with st.spinner(""):
        # Show loading animation
        st.markdown("<div class='loading-animation'><div class='netflix-loader'></div></div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #E6E6E6;'>Analyzing your review... 🎭</p>", unsafe_allow_html=True)
        
        # Simulate processing time
        time.sleep(2)
        
        # Load model and make prediction (replace with actual implementation)
        model, word_index = load_sentiment_model()
        sentiment, score = predict_sentiment(user_input, model, word_index)
        
        # Clear loading
        st.empty()
        
        # Display results with animations
        col1, col2 = st.columns(2)
        
        with col1:
            if sentiment == "Positive":
                st.markdown(f"""
                <div class='sentiment-positive'>
                    <h2>🎉 POSITIVE REVIEW!</h2>
                    <p>This review expresses positive sentiment about the movie</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='sentiment-negative'>
                    <h2>👎 NEGATIVE REVIEW</h2>
                    <p>This review expresses negative sentiment about the movie</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            confidence_percentage = int(score * 100) if sentiment == "Positive" else int((1 - score) * 100)
            fill_color = "#00b09b" if sentiment == "Positive" else "#E50914"
            
            st.markdown(f"""
            <div style='text-align: center;'>
                <h3>Confidence Score</h3>
                <div class='prediction-score'>{confidence_percentage}%</div>
                <div class='confidence-bar'>
                    <div class='confidence-fill' style='width: {confidence_percentage}%; background: {fill_color};'></div>
                </div>
                <p style='color: #E6E6E6; margin-top: 1rem;'>
                    The model is {confidence_percentage}% confident this review is {sentiment.lower()}
                </p>
            </div>
            """, unsafe_allow_html=True)

elif analyze_clicked and not user_input.strip():
    st.warning("⚠️ Please enter a movie review to analyze!")

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

with footer_col2:
    st.markdown(
        """
        <div class='footer'>
            <div style='text-align: center;'>
                <h4 style='color: #E50914; margin-bottom: 1rem; font-size: 1.5rem;'>👨‍💻 Developed By</h4>
                <p style='font-size: 1.4rem; font-weight: bold; color: #FFFFFF; margin-bottom: 0.5rem;'>
                    Harshith Narasimhamurthy
                </p>
                <p style='margin-bottom: 0.5rem; font-size: 1.1rem; color: #E6E6E6;'>
                    📧 harshithnchandan@gmail.com | 📱 +919663918804
                </p>
                <p style='margin-bottom: 1rem; font-size: 1.1rem;'>
                    🔗 <a href='https://www.linkedin.com/in/harshithnarasimhamurthy69/' target='_blank' 
                       style='color: #E50914; text-decoration: none; font-weight: bold;'>
                       Connect with me on LinkedIn
                    </a>
                </p>
                <p style='font-size: 1rem; color: rgba(255,255,255,0.8);'>
                    Powered by TensorFlow & Streamlit | Netflix-Themed Analytics Dashboard
                </p>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Add some cinematic elements
st.markdown("""
<div style='text-align: center; margin-top: 2rem;'>
    <p style='color: #666; font-size: 0.9rem;'>
        🎬 Lights, Camera, Sentiment Analysis! 🎭
    </p>
</div>
""", unsafe_allow_html=True)
