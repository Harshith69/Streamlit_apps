import streamlit as st
import pypdf
import tempfile
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle

def initialize_session_state():
    if 'resume_processed' not in st.session_state:
        st.session_state.resume_processed = False
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = ""
    if 'resume_chunks' not in st.session_state:
        st.session_state.resume_chunks = []
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ""
    if 'first_interaction' not in st.session_state:
        st.session_state.first_interaction = True
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = None
    if 'faiss_index' not in st.session_state:
        st.session_state.faiss_index = None

@st.cache_resource
def load_embedding_model():
    """Load lightweight embedding model"""
    try:
        # Use a very small, efficient model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        return None

def extract_personal_info(text):
    """Extract name from resume text"""
    name_patterns = [
        r'^([A-Z][a-z]+ [A-Z][a-z]+)',
        r'\n([A-Z][a-z]+ [A-Z][a-z]+)\s*\n',
        r'Name[:]?\s*([A-Z][a-z]+ [A-Z][a-z]+)',
        r'([A-Z][a-z]+ [A-Z][a-z]+)\s+[\w\.-]+@[\w\.-]+',
    ]
    
    name = "the candidate"
    for pattern in name_patterns:
        match = re.search(pattern, text)
        if match:
            name = match.group(1)
            break
    
    if name == "the candidate":
        lines = text.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+$', line):
                name = line
                break
    
    return {'name': name}

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF"""
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
        
        os.unlink(tmp_path)
        return text
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise e

def create_smart_chunks(text, max_chunk_size=400):
    """Create intelligent text chunks preserving context"""
    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph exceeds chunk size, save current chunk
        if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += " " + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # If chunks are too large, split by sentences
    if not chunks or any(len(chunk) > max_chunk_size * 1.5 for chunk in chunks):
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += ". " + sentence + "."
                else:
                    current_chunk = sentence + "."
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    return chunks

def create_embeddings(chunks, model):
    """Create embeddings for text chunks"""
    if not chunks or not model:
        return None
    
    try:
        embeddings = model.encode(chunks)
        return embeddings
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None

def create_faiss_index(embeddings):
    """Create FAISS index for efficient similarity search"""
    if embeddings is None:
        return None
    
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product index for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        return index
    except Exception as e:
        st.error(f"Error creating FAISS index: {str(e)}")
        return None

def find_similar_chunks(question, model, index, chunks, top_k=3):
    """Find most relevant chunks using FAISS"""
    if index is None or not chunks:
        return []
    
    try:
        # Encode question
        question_embedding = model.encode([question])
        faiss.normalize_L2(question_embedding)
        
        # Search
        scores, indices = index.search(question_embedding, top_k)
        
        # Return relevant chunks
        relevant_chunks = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks):
                relevant_chunks.append(chunks[idx])
        
        return relevant_chunks
    except Exception as e:
        st.error(f"Error in similarity search: {str(e)}")
        return []

def get_conversation_context():
    """Get recent conversation history"""
    if not st.session_state.conversation_history:
        return ""
    
    context = "Recent conversation:\n"
    for i, (question, answer) in enumerate(st.session_state.conversation_history[-3:]):
        context += f"Q: {question}\nA: {answer}\n"
    
    return context

def generate_smart_response(question, relevant_chunks):
    """Generate response based on relevant chunks and conversation history"""
    if not relevant_chunks:
        return "I don't have specific information about that in my resume. Please ask about my skills, experience, education, or projects."
    
    # Combine relevant chunks
    context = "\n".join(relevant_chunks)
    conversation_history = get_conversation_context()
    
    question_lower = question.lower()
    
    # Enhanced response templates based on question type
    if any(word in question_lower for word in ['hello', 'hi', 'hey', 'introduce']):
        response = f"Hello! I'm {st.session_state.user_name}. {context[:200]}... I'd be happy to discuss my background in more detail. What would you like to know?"
    
    elif any(word in question_lower for word in ['skill', 'technology', 'programming']):
        response = f"Based on my resume, here are my relevant skills:\n\n{context}\n\nI'm always learning and expanding my technical capabilities."
    
    elif any(word in question_lower for word in ['experience', 'work', 'job', 'role']):
        response = f"Regarding my professional experience:\n\n{context}\n\nThis experience has helped me develop strong capabilities in my field."
    
    elif any(word in question_lower for word in ['education', 'degree', 'university', 'college']):
        response = f"About my educational background:\n\n{context}\n\nMy education provides a solid foundation for my professional work."
    
    elif any(word in question_lower for word in ['project', 'portfolio']):
        response = f"Here are some projects I've worked on:\n\n{context}\n\nThese projects demonstrate my practical skills and problem-solving abilities."
    
    elif any(word in question_lower for word in ['achievement', 'accomplishment']):
        response = f"Some of my key achievements include:\n\n{context}\n\nI'm proud of these accomplishments and the value they've delivered."
    
    else:
        response = f"Regarding your question:\n\n{context}\n\nIs there anything specific you'd like me to elaborate on?"
    
    return response

def process_resume(pdf_file):
    """Process resume with embeddings and FAISS"""
    try:
        # Extract text
        text = extract_text_from_pdf(pdf_file)
        
        if not text.strip():
            raise ValueError("No readable text found in PDF")
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        st.session_state.resume_text = text
        
        # Extract personal info
        personal_info = extract_personal_info(text)
        st.session_state.user_name = personal_info['name']
        
        # Create chunks
        chunks = create_smart_chunks(text)
        st.session_state.resume_chunks = chunks
        
        # Load embedding model
        with st.spinner("Loading AI model..."):
            model = load_embedding_model()
            if not model:
                raise Exception("Failed to load embedding model")
            st.session_state.embedding_model = model
        
        # Create embeddings
        with st.spinner("Processing resume content..."):
            embeddings = create_embeddings(chunks, model)
            if embeddings is None:
                raise Exception("Failed to create embeddings")
            st.session_state.embeddings = embeddings
        
        # Create FAISS index
        with st.spinner("Building search index..."):
            faiss_index = create_faiss_index(embeddings)
            if not faiss_index:
                raise Exception("Failed to create search index")
            st.session_state.faiss_index = faiss_index
        
        st.success(f"✅ Resume processed! Welcome {st.session_state.user_name}")
        return True
        
    except Exception as e:
        raise e

def generate_response(question):
    """Generate response using embeddings and FAISS"""
    try:
        if (not st.session_state.embedding_model or 
            not st.session_state.faiss_index or 
            not st.session_state.resume_chunks):
            return "System not ready. Please re-upload your resume."
        
        # Find relevant chunks
        relevant_chunks = find_similar_chunks(
            question,
            st.session_state.embedding_model,
            st.session_state.faiss_index,
            st.session_state.resume_chunks
        )
        
        # Generate response
        response = generate_smart_response(question, relevant_chunks)
        
        # Update conversation history
        st.session_state.conversation_history.append((question, response))
        
        # Keep only last 10 conversations to manage memory
        if len(st.session_state.conversation_history) > 10:
            st.session_state.conversation_history = st.session_state.conversation_history[-10:]
        
        return response
        
    except Exception as e:
        return f"I apologize, but I encountered an issue. Please try again or rephrase your question."

# UI Code
st.set_page_config(page_title="AI Resume Assistant", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
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
    padding: 1.5rem;
    border-radius: 10px;
    margin-top: 2rem;
}
.welcome-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    color: white;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

def main():
    initialize_session_state()
    
    st.markdown('<div class="main-header">🤖 AI Resume Assistant</div>', unsafe_allow_html=True)
    
    if not st.session_state.resume_processed:
        st.markdown("### Upload your resume and chat with AI!")
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
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.markdown(f"""
        <div class="welcome-banner">
            <h3>👋 Hello! I'm {st.session_state.user_name}</h3>
            <p>AI-Powered Resume Assistant | Ready to discuss my experience</p>
            <p><small>Powered by Sentence Transformers & FAISS</small></p>
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
                "What are your key achievements?"
            ]
            
            for example in examples:
                if st.button(example, use_container_width=True):
                    if len(st.session_state.messages) == 0 and st.session_state.first_interaction:
                        welcome_msg = f"Hello! I'm {st.session_state.user_name}. Thank you for reviewing my resume! I'd be happy to discuss my experience, skills, and background. How can I assist you today?"
                        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
                        st.session_state.first_interaction = False
                    
                    st.session_state.messages.append({"role": "user", "content": example})
                    response = generate_response(example)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
            
            st.markdown("---")
            if st.button("🔄 Upload New Resume", use_container_width=True):
                # Reset session state
                st.session_state.resume_processed = False
                st.session_state.messages = []
                st.session_state.resume_text = ""
                st.session_state.resume_chunks = []
                st.session_state.user_name = ""
                st.session_state.first_interaction = True
                st.session_state.conversation_history = []
                st.session_state.embedding_model = None
                st.session_state.faiss_index = None
                st.session_state.embeddings = None
                st.rerun()
            
            st.markdown("---")
            st.markdown("### 🔧 System Info")
            st.info("Using: Sentence Transformers + FAISS + Smart Memory")
        
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
            response = generate_response(user_question)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div class='footer'>
            <div style='text-align: center;'>
                <h4 style='color: #E50914; margin-bottom: 1rem; font-size: 1.2rem;'>👨‍💻 Developed By</h4>
                <p style='font-size: 1.1rem; font-weight: bold; color: #FFFFFF; margin-bottom: 0.5rem;'>
                    Harshith Narasimhamurthy
                </p>
                <p style='margin-bottom: 0.5rem; font-size: 0.9rem; color: #E6E6E6;'>
                    📧 harshithnchandan@gmail.com
                </p>
                <p style='margin-bottom: 1rem; font-size: 0.9rem;'>
                    🔗 <a href='https://www.linkedin.com/in/harshithnarasimhamurthy69/' target='_blank' 
                       style='color: #E50914; text-decoration: none; font-weight: bold;'>
                       LinkedIn
                    </a>
                </p>
                <p style='font-size: 0.8rem; color: rgba(255,255,255,0.8);'>
                    Lightweight AI Resume Assistant
                </p>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
