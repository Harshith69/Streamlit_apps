import streamlit as st
import pypdf
import tempfile
import os
import re

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
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None

def load_dialogpt_model():
    """Load DialoGPT-large only when needed"""
    try:
        # Import inside function to avoid initial loading issues
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
        
        st.info("🚀 Loading DialoGPT-large model...")
        model_name = "microsoft/DialoGPT-large"
        
        # Load with specific settings for compatibility
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for compatibility
            low_cpu_mem_usage=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        chatbot = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,  # Reduced for stability
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        
        st.session_state.model_loaded = True
        return chatbot
        
    except Exception as e:
        st.error(f"❌ Failed to load DialoGPT-large: {str(e)}")
        st.info("Using enhanced rule-based system as fallback")
        return None

def extract_personal_info(text):
    """Extract name and basic info from resume"""
    name_patterns = [
        r'^([A-Z][a-z]+ [A-Z][a-z]+)',
        r'([A-Z][a-z]+ [A-Z][a-z]+)\s*\n',
        r'Name[:]?\s*([A-Z][a-z]+ [A-Z][a-z]+)'
    ]
    
    name = "the candidate"
    for pattern in name_patterns:
        match = re.search(pattern, text)
        if match:
            name = match.group(1)
            break
    
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, text)
    email = email_match.group() if email_match else ""
    
    return {'name': name, 'email': email}

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
        personal_info = extract_personal_info(text)
        st.session_state.user_name = personal_info['name']
        st.session_state.personal_info = personal_info
        
        os.unlink(tmp_path)
        return True
        
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise e

def get_conversation_history():
    """Get conversation context for AI"""
    if len(st.session_state.messages) == 0:
        return ""
    
    history = "Previous conversation:\n"
    for msg in st.session_state.messages[-4:]:
        speaker = "Human" if msg["role"] == "user" else st.session_state.user_name
        history += f"{speaker}: {msg['content']}\n"
    return history

def extract_relevant_context(question, resume_text):
    """Extract relevant resume content"""
    question_lower = question.lower()
    sentences = re.split(r'[.!?]+', resume_text)
    scored_sentences = []
    
    for sentence in sentences:
        if len(sentence.strip()) < 20:
            continue
            
        sentence_lower = sentence.lower()
        score = 0
        
        question_words = set(question_lower.split())
        sentence_words = set(sentence_lower.split())
        common_words = question_words.intersection(sentence_words)
        score += len(common_words) * 3
        
        if score > 0:
            scored_sentences.append((score, sentence.strip()))
    
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    relevant_sentences = [sentence for score, sentence in scored_sentences[:5]]
    
    return ". ".join(relevant_sentences) if relevant_sentences else "No specific information available."

def generate_dialogpt_response(question, resume_text):
    """Generate response using DialoGPT-large"""
    try:
        # Load model if not already loaded
        if st.session_state.chatbot is None:
            st.session_state.chatbot = load_dialogpt_model()
        
        if st.session_state.chatbot is None:
            return None  # Fallback to rule-based
        
        context = extract_relevant_context(question, resume_text)
        conversation_history = get_conversation_history()
        
        prompt = f"""You are {st.session_state.user_name}. Answer based on resume context.

RESUME CONTEXT:
{context}

{conversation_history}

Human: {question}

{st.session_state.user_name}:"""
        
        with st.spinner('🤔 Thinking with AI...'):
            responses = st.session_state.chatbot(
                prompt,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=st.session_state.chatbot.tokenizer.eos_token_id
            )
        
        if responses:
            generated_text = responses[0]['generated_text']
            ai_response = generated_text.replace(prompt, "").strip()
            # Clean response
            ai_response = re.split(r'[.!?]', ai_response)[0] + '.'
            return ai_response
        
    except Exception as e:
        st.error(f"AI Error: {str(e)}")
    
    return None

def generate_fallback_response(question, resume_text):
    """Enhanced rule-based fallback"""
    context = extract_relevant_context(question, resume_text)
    
    if "no specific information" in context.lower():
        return f"I apologize, but I don't have specific information about '{question}' in my resume. I'd be happy to discuss my skills, experience, education, or projects instead."
    
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['skill', 'technology', 'programming']):
        return f"Based on my resume, here are my relevant skills:\n\n{context}\n\nI'm continuously learning and expanding my technical expertise."
    
    elif any(word in question_lower for word in ['experience', 'work', 'job']):
        return f"Regarding my professional experience:\n\n{context}\n\nThis experience has helped me develop valuable insights and capabilities."
    
    elif any(word in question_lower for word in ['education', 'degree', 'university']):
        return f"About my educational background:\n\n{context}\n\nMy education has provided a strong foundation for my career growth."
    
    elif any(word in question_lower for word in ['project', 'portfolio']):
        return f"Here are some projects from my experience:\n\n{context}\n\nThese projects have been instrumental in developing my skills."
    
    elif any(word in question_lower for word in ['advice', 'suggest', 'improve']):
        return f"Based on my journey, I'd suggest focusing on continuous learning and practical application. From my experience: {context}"
    
    else:
        return f"Regarding your question:\n\n{context}\n\nIs there anything specific you'd like me to elaborate on?"

def generate_response(question, resume_text):
    """Try AI first, then fallback"""
    # Try DialoGPT first
    ai_response = generate_dialogpt_response(question, resume_text)
    if ai_response:
        return ai_response
    
    # Fallback to enhanced rule-based
    return generate_fallback_response(question, resume_text)

# Rest of your Streamlit UI code remains the same...
# [Include all your existing UI code with the footer]

st.set_page_config(page_title="AI Resume Assistant", page_icon="🤖", layout="wide")

# [Include all your CSS and UI code from previous version]
# [Include the main() function and footer from previous version]

def main():
    initialize_session_state()
    
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
        st.markdown("### Upload your resume and chat with DialoGPT-powered AI!")
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        st.markdown("### 📄 Upload Your Resume (PDF only)")
        
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed")
        
        if uploaded_file is not None:
            st.info(f"📁 File: {uploaded_file.name}")
            
            if st.button("🚀 Process Resume", type="primary"):
                with st.spinner('Processing your resume...'):
                    try:
                        if process_resume(uploaded_file):
                            st.session_state.resume_processed = True
                            st.success("✅ Resume processed successfully!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.markdown(f"""
        <div class="welcome-banner">
            <h3>👋 Hello! I'm {st.session_state.user_name}</h3>
            <p>Powered by DialoGPT-large AI | Ready to discuss my experience and background</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.sidebar:
            st.markdown("### 💡 Conversation Starters")
            examples = [
                "Can you introduce yourself?",
                "What are your technical skills?",
                "Tell me about your work experience",
                "What projects have you worked on?",
                "What's your educational background?",
                "Can you give career advice?"
            ]
            
            for example in examples:
                if st.button(example, use_container_width=True):
                    if len(st.session_state.messages) == 0 and st.session_state.first_interaction:
                        welcome_msg = f"Hello! I'm {st.session_state.user_name}. Thank you for reviewing my resume! I'd be happy to discuss my experience, skills, and background. How can I assist you today?"
                        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
                        st.session_state.first_interaction = False
                    
                    st.session_state.messages.append({"role": "user", "content": example})
                    response = generate_response(example, st.session_state.resume_text)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
            
            st.markdown("---")
            if st.button("🔄 Upload New Resume", use_container_width=True):
                st.session_state.resume_processed = False
                st.session_state.messages = []
                st.session_state.resume_text = ""
                st.session_state.user_name = ""
                st.session_state.first_interaction = True
                st.session_state.model_loaded = False
                st.session_state.chatbot = None
                st.rerun()
        
        st.markdown("### 💬 Conversation")
        
        if len(st.session_state.messages) == 0 and st.session_state.first_interaction:
            welcome_msg = f"Hello! I'm {st.session_state.user_name}. Thank you for reviewing my resume! I'd be happy to discuss my experience, skills, and background. How can I assist you today?"
            st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
            st.session_state.first_interaction = False
        
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
        
        if user_question := st.chat_input(f"Ask {st.session_state.user_name}..."):
            st.session_state.messages.append({"role": "user", "content": user_question})
            response = generate_response(user_question, st.session_state.resume_text)
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
