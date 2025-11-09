import streamlit as st
import pypdf
import tempfile
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def initialize_session_state():
    if 'resume_processed' not in st.session_state:
        st.session_state.resume_processed = False
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = ""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ""
    if 'first_interaction' not in st.session_state:
        st.session_state.first_interaction = True
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None

@st.cache_resource
def load_dialogpt_model():
    """Load DialoGPT-large model"""
    try:
        st.info("🚀 Loading DialoGPT-large model... This may take a minute.")
        model_name = "microsoft/DialoGPT-large"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        chatbot = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            max_length=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        
        st.success("✅ DialoGPT-large loaded successfully!")
        return chatbot
    except Exception as e:
        st.error(f"❌ Failed to load DialoGPT: {str(e)}")
        return None

def extract_personal_info(text):
    """Extract name and basic info from resume"""
    name_pattern = r'^([A-Z][a-z]+ [A-Z][a-z]+)'
    name_match = re.search(name_pattern, text)
    name = name_match.group(1) if name_match else "the candidate"
    
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, text)
    email = email_match.group() if email_match else ""
    
    return {
        'name': name,
        'email': email
    }

def process_resume(pdf_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        with open(tmp_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text:
            raise ValueError("No readable text found in PDF")
        
        st.session_state.resume_text = text
        
        # Extract personal info
        personal_info = extract_personal_info(text)
        st.session_state.user_name = personal_info['name']
        st.session_state.personal_info = personal_info
        
        # Load AI model
        st.session_state.chatbot = load_dialogpt_model()
        
        os.unlink(tmp_path)
        return True
        
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise e

def get_conversation_history():
    """Get formatted conversation history"""
    if len(st.session_state.messages) == 0:
        return ""
    
    history = "Previous conversation:\n"
    for msg in st.session_state.messages[-6:]:  # Last 3 exchanges
        speaker = "Human" if msg["role"] == "user" else "AI"
        history += f"{speaker}: {msg['content']}\n"
    return history

def extract_relevant_context(question, resume_text):
    """Extract relevant context from resume based on question"""
    question_lower = question.lower()
    
    sentences = re.split(r'[.!?]+', resume_text)
    scored_sentences = []
    
    for sentence in sentences:
        if len(sentence.strip()) < 20:
            continue
            
        sentence_lower = sentence.lower()
        score = 0
        
        # Keyword matching
        question_words = set(question_lower.split())
        sentence_words = set(sentence_lower.split())
        common_words = question_words.intersection(sentence_words)
        score += len(common_words) * 3
        
        if score > 0:
            scored_sentences.append((score, sentence.strip()))
    
    # Get top relevant sentences
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    relevant_sentences = [sentence for score, sentence in scored_sentences[:5]]
    
    return ". ".join(relevant_sentences) if relevant_sentences else "No specific information available about this topic."

def generate_ai_response(question, resume_text):
    """Generate response using DialoGPT-large"""
    try:
        if st.session_state.chatbot is None:
            return "AI model is not available. Please try again."
        
        # Extract relevant context
        context = extract_relevant_context(question, resume_text)
        conversation_history = get_conversation_history()
        
        # Build intelligent prompt for DialoGPT
        prompt = f"""You are {st.session_state.user_name}, a professional candidate. Answer questions based on your resume and experience.

RESUME CONTEXT:
{context}

{conversation_history}

Human: {question}

{st.session_state.user_name}:"""
        
        # Generate response using DialoGPT
        with st.spinner('🤔 Thinking...'):
            responses = st.session_state.chatbot(
                prompt,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=st.session_state.chatbot.tokenizer.eos_token_id
            )
        
        if responses and len(responses) > 0:
            generated_text = responses[0]['generated_text']
            # Extract only the AI's response (after the prompt)
            ai_response = generated_text.replace(prompt, "").strip()
            
            # Clean up the response
            ai_response = re.split(r'[.!?]', ai_response)[0] + '.'  # Take first complete sentence
            ai_response = ai_response.split('\n')[0]  # Take first line
            
            return ai_response if ai_response else "I'd be happy to discuss my experience. What would you like to know?"
        
        return "I'd be happy to discuss my experience. What would you like to know?"
        
    except Exception as e:
        return f"I apologize, but I encountered an issue: {str(e)}. Please try asking again."

def get_welcome_message():
    """Generate personalized welcome message"""
    if st.session_state.first_interaction:
        st.session_state.first_interaction = False
        return f"Hello! I'm {st.session_state.user_name}. Thank you for taking the time to review my resume! I'm excited to discuss my experience, skills, and background. What would you like to know about my professional journey?"

def main():
    initialize_session_state()
    
    st.set_page_config(page_title="AI Resume Assistant", page_icon="🤖", layout="wide")
    
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .upload-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        color: white;
    }
    .chat-message {
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border-left: 5px solid;
    }
    .user-message {
        background: rgba(74, 144, 226, 0.15);
        border-left-color: #4a90e2;
        margin-left: 2rem;
    }
    .assistant-message {
        background: rgba(46, 204, 113, 0.15);
        border-left-color: #2ecc71;
        margin-right: 2rem;
    }
    .footer {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
    }
    .welcome-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">🤖 AI Resume Assistant</div>', unsafe_allow_html=True)
    
    if not st.session_state.resume_processed:
        st.markdown("### Upload your resume and chat with an AI version of yourself!")
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        st.markdown("### 📄 Upload Your Resume (PDF only)")
        
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed")
        
        if uploaded_file is not None:
            st.info(f"📁 File: {uploaded_file.name}")
            
            if st.button("🚀 Process Resume", type="primary"):
                with st.spinner('Processing your resume and loading AI model...'):
                    try:
                        if process_resume(uploaded_file):
                            st.session_state.resume_processed = True
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Welcome banner
        st.markdown(f"""
        <div class="welcome-banner">
            <h3>👋 Hello! I'm {st.session_state.user_name}</h3>
            <p>Powered by DialoGPT-large AI | I'm here to discuss my experience and background</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.sidebar:
            st.markdown("### 💡 Conversation Starters")
            examples = [
                "Can you introduce yourself professionally?",
                "What are your strongest technical skills?",
                "Tell me about your work experience",
                "What projects are you most proud of?",
                "How do you approach problem-solving?",
                "What are your career goals?",
                "Can you give me career advice?"
            ]
            
            for example in examples:
                if st.button(example, use_container_width=True):
                    if len(st.session_state.messages) == 0:
                        welcome_msg = get_welcome_message()
                        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
                    
                    st.session_state.messages.append({"role": "user", "content": example})
                    response = generate_ai_response(example, st.session_state.resume_text)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
            
            st.markdown("---")
            if st.button("🔄 Upload New Resume", use_container_width=True):
                st.session_state.resume_processed = False
                st.session_state.messages = []
                st.session_state.resume_text = ""
                st.session_state.user_name = ""
                st.session_state.first_interaction = True
                st.session_state.chatbot = None
                st.rerun()
        
        # Display chat
        st.markdown("### 💬 Conversation")
        
        # Show welcome message if first interaction
        if len(st.session_state.messages) == 0:
            welcome_msg = get_welcome_message()
            st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
        
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="chat-message user-message">'
                    f'<strong>👤 You:</strong><br>{message["content"]}'
                    f'</div>', 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-message assistant-message">'
                    f'<strong>🤖 {st.session_state.user_name}:</strong><br>{message["content"]}'
                    f'</div>', 
                    unsafe_allow_html=True
                )
        
        # Chat input
        if user_question := st.chat_input(f"Ask {st.session_state.user_name}..."):
            st.session_state.messages.append({"role": "user", "content": user_question})
            response = generate_ai_response(user_question, st.session_state.resume_text)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    # Footer
    st.markdown("---")
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
                    Powered by DialoGPT-large & Streamlit | Intelligent Resume Assistant
                </p>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
