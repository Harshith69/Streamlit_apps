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

def simple_search(question, text):
    question_lower = question.lower()
    text_lower = text.lower()
    
    # Simple keyword matching
    question_words = set(question_lower.split())
    text_words = set(text_lower.split())
    
    common_words = question_words.intersection(text_words)
    
    if common_words:
        # Find sentences containing matching words
        sentences = text.split('. ')
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in common_words):
                relevant_sentences.append(sentence)
        
        return ". ".join(relevant_sentences[:5]) + "."
    else:
        return "No specific information found about this topic in the resume."

def generate_response(question, context):
    if context and context != "No specific information found about this topic in the resume.":
        return f"Based on my resume:\n\n{context}"
    else:
        return "I don't have specific information about that in my resume. Try asking about my skills, experience, education, or projects."

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
                "What are my skills?",
                "What is my work experience?",
                "Tell me about my education",
                "What projects have I done?"
            ]
            
            for example in examples:
                if st.button(example, use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": example})
                    context = simple_search(example, st.session_state.resume_text)
                    response = generate_response(example, context)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
            
            if st.button("🔄 Upload New Resume", use_container_width=True):
                st.session_state.resume_processed = False
                st.session_state.messages = []
                st.session_state.resume_text = ""
                st.rerun()
        
        st.markdown("### 💬 Chat with Your Resume")
        
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")
        
        if user_question := st.chat_input("Ask about your resume..."):
            st.session_state.messages.append({"role": "user", "content": user_question})
            context = simple_search(user_question, st.session_state.resume_text)
            response = generate_response(user_question, context)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

if __name__ == "__main__":
    main()
