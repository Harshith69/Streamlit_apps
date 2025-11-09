import streamlit as st
import pypdf
import tempfile
import os
import re
import requests
import json

def initialize_session_state():
    if 'resume_processed' not in st.session_state:
        st.session_state.resume_processed = False
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = ""
    if 'messages' not in st.session_state:
        st.session_state.messages = []

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
        os.unlink(tmp_path)
        return True
        
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise e

def query_huggingface_model(context, question):
    """Use Hugging Face Inference API for better responses"""
    try:
        API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large"
        headers = {"Authorization": "Bearer hf_your_token_here"}  # You can get free token from huggingface.co
        
        # Create a better prompt
        prompt = f"""
        Based on the following resume information, answer the question professionally and accurately.
        
        RESUME CONTEXT:
        {context[:2000]}
        
        QUESTION: {question}
        
        INSTRUCTIONS:
        - Only use information from the resume context above
        - If the information isn't in the resume, politely say so
        - Be professional, concise, and helpful
        - Format the response clearly with bullet points if appropriate
        
        ANSWER:
        """
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.3,
                "do_sample": True
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and 'generated_text' in result[0]:
                return result[0]['generated_text'].replace(prompt, "").strip()
        return None
    except:
        return None

def get_ai_response(question, resume_text):
    """Get intelligent response using better context matching"""
    # Improved context extraction
    question_lower = question.lower()
    resume_lower = resume_text.lower()
    
    # Extract relevant sections based on question type
    sections = {
        'skills': ['skill', 'technology', 'programming', 'framework', 'tool', 'language'],
        'experience': ['experience', 'work', 'job', 'employment', 'role', 'position'],
        'education': ['education', 'degree', 'university', 'college', 'course', 'qualification'],
        'projects': ['project', 'portfolio', 'work', 'development', 'built', 'created']
    }
    
    # Find which section the question belongs to
    question_type = None
    for section, keywords in sections.items():
        if any(keyword in question_lower for keyword in keywords):
            question_type = section
            break
    
    # Extract relevant sentences with better scoring
    sentences = re.split(r'[.!?]+', resume_text)
    scored_sentences = []
    
    for sentence in sentences:
        if len(sentence.strip()) < 15:
            continue
            
        sentence_lower = sentence.lower()
        score = 0
        
        # Score based on keyword matches
        question_words = set(question_lower.split())
        sentence_words = set(sentence_lower.split())
        common_words = question_words.intersection(sentence_words)
        score += len(common_words) * 3
        
        # Bonus for section-specific keywords
        if question_type and any(keyword in sentence_lower for keyword in sections[question_type]):
            score += 5
            
        # Bonus for exact phrase matches
        for word in question_words:
            if len(word) > 3 and word in sentence_lower:
                score += 2
        
        if score > 0:
            scored_sentences.append((score, sentence.strip()))
    
    # Get top relevant sentences
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    relevant_sentences = [sentence for score, sentence in scored_sentences[:5]]
    
    if relevant_sentences:
        context = ". ".join(relevant_sentences) + "."
        
        # Try to use Hugging Face model for better response
        ai_response = query_huggingface_model(context, question)
        if ai_response:
            return ai_response
        
        # Fallback to smart formatting
        return format_smart_response(question, context)
    else:
        return "I don't have specific information about that in my resume. Please try asking about:\n\n• My skills and technologies\n• Work experience\n• Education background\n• Projects I've worked on"

def format_smart_response(question, context):
    """Format responses intelligently based on question type"""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['skill', 'technology', 'programming']):
        return f"Based on my resume, here are my key skills and technologies:\n\n{context}"
    
    elif any(word in question_lower for word in ['experience', 'work', 'job']):
        return f"Based on my resume, here's my relevant work experience:\n\n{context}"
    
    elif any(word in question_lower for word in ['education', 'degree', 'university']):
        return f"Based on my resume, here's my educational background:\n\n{context}"
    
    elif any(word in question_lower for word in ['project', 'portfolio']):
        return f"Based on my resume, here are the projects I've worked on:\n\n{context}"
    
    else:
        return f"Based on my resume:\n\n{context}"

def simple_search(question, text):
    """Improved search with better context extraction"""
    return get_ai_response(question, text)

def generate_response(question, context):
    """Generate response using the improved AI"""
    return get_ai_response(question, context)

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
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    border-left: 4px solid;
}
.user-message {
    background: rgba(74, 144, 226, 0.1);
    border-left-color: #4a90e2;
}
.assistant-message {
    background: rgba(46, 204, 113, 0.1);
    border-left-color: #2ecc71;
}
.footer {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 2rem;
    border-radius: 15px;
    margin-top: 2rem;
    border: 1px solid rgba(255,255,255,0.1);
}
</style>
""", unsafe_allow_html=True)

def main():
    initialize_session_state()
    
    st.markdown('<div class="main-header">🤖 AI Resume Assistant</div>', unsafe_allow_html=True)
    st.markdown("### Upload your resume and chat with your AI assistant!")
    
    if not st.session_state.resume_processed:
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
        with st.sidebar:
            st.markdown("### 💡 Example Questions")
            examples = [
                "What are my technical skills?",
                "Summarize my work experience",
                "Tell me about my education",
                "What projects have I completed?",
                "What programming languages do I know?",
                "Describe my professional background"
            ]
            
            for example in examples:
                if st.button(example, use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": example})
                    response = generate_response(example, st.session_state.resume_text)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
            
            if st.button("🔄 Upload New Resume", use_container_width=True):
                st.session_state.resume_processed = False
                st.session_state.messages = []
                st.session_state.resume_text = ""
                st.rerun()
        
        st.markdown("### 💬 Chat with Your Resume")
        
        # Display chat messages with better formatting
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
                    f'<strong>🤖 Assistant:</strong><br>{message["content"]}'
                    f'</div>', 
                    unsafe_allow_html=True
                )
        
        if user_question := st.chat_input("Ask about your resume..."):
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
                    Powered by AI & Streamlit | Intelligent Resume Analysis
                </p>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
