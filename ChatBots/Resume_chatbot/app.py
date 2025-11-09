import streamlit as st
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
import tempfile
import os
import re

# Page configuration
st.set_page_config(
    page_title="AI Resume Assistant 🤖",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for vibrant design
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

.chat-container {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
}

.user-message {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 1rem;
    border-radius: 12px;
    margin: 0.5rem 0;
    color: white;
    border: 1px solid rgba(255,255,255,0.3);
}

.assistant-message {
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    padding: 1rem;
    border-radius: 12px;
    margin: 0.5rem 0;
    color: white;
    border: 1px solid rgba(255,255,255,0.3);
}

.stButton button {
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    color: white;
    border: none;
    padding: 0.5rem 1.5rem;
    border-radius: 20px;
    font-weight: bold;
    margin: 0.5rem;
}

.sidebar-content {
    background: rgba(255,255,255,0.1);
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'resume_processed' not in st.session_state:
        st.session_state.resume_processed = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'collection' not in st.session_state:
        st.session_state.collection = None
    if 'embedder' not in st.session_state:
        st.session_state.embedder = None

def clean_text(text):
    """Clean and preprocess text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:]', '', text)
    return text.strip()

def process_resume(pdf_file):
    """Process uploaded PDF resume"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        # Read PDF
        with open(tmp_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        # Clean text
        text = clean_text(text)
        
        if not text:
            raise ValueError("No readable text found in PDF")
        
        # Split into chunks
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < 800:  # Smaller chunks for better performance
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Initialize embedder
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create ChromaDB client and collection
        client = chromadb.Client()
        collection = client.create_collection("resume_data")
        
        # Add documents in batches to avoid memory issues
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embedder.encode(batch_chunks).tolist()
            
            # Add to collection
            for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    ids=[f"chunk_{i+j}"]
                )
        
        # Clean up
        os.unlink(tmp_path)
        
        return collection, embedder
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise e

def get_similar_content(question, top_k=3):
    """Find similar content from resume"""
    try:
        query_embedding = st.session_state.embedder.encode([question]).tolist()
        results = st.session_state.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        if results['documents'] and results['documents'][0]:
            return " ".join(results['documents'][0])
        return "No relevant information found in resume."
        
    except Exception as e:
        return f"Error searching resume: {str(e)}"

def generate_response(question, context):
    """Generate response based on context"""
    prompt = f"""
    Based on the following resume information, answer the question.
    
    RESUME CONTEXT:
    {context}
    
    QUESTION: {question}
    
    INSTRUCTIONS:
    - Only use information from the resume context above
    - If the information isn't in the resume, say "This information is not in my resume"
    - Be concise and professional
    - Format your response clearly
    
    ANSWER:
    """
    
    # Simple response generation (you can enhance this later)
    if len(context) > 50:  # If we have meaningful context
        return f"Based on my resume:\n\n{context[:400]}..."
    else:
        return "I couldn't find relevant information about that in my resume. Please try asking about my skills, experience, education, or projects."

def main():
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🤖 AI Resume Assistant</div>', unsafe_allow_html=True)
    st.markdown("### Upload your resume and chat with your AI assistant!")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 💡 How to Use")
        st.markdown("""
        1. **Upload** your resume (PDF)
        2. **Wait** for processing
        3. **Ask questions** about your experience
        
        **Example questions:**
        - What are my skills?
        - Summarize my experience
        - What projects have I done?
        - Tell me about my education
        """)
        
        if st.session_state.resume_processed:
            if st.button("🔄 Upload New Resume", use_container_width=True):
                st.session_state.resume_processed = False
                st.session_state.messages = []
                st.session_state.collection = None
                st.session_state.embedder = None
                st.rerun()
    
    # Main content
    if not st.session_state.resume_processed:
        # Upload section
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        st.markdown("### 📄 Upload Your Resume")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            st.info(f"📁 File: {uploaded_file.name} | Size: {uploaded_file.size // 1024} KB")
            
            if st.button("🚀 Process Resume", type="primary"):
                with st.spinner('Processing your resume... This may take a moment.'):
                    try:
                        collection, embedder = process_resume(uploaded_file)
                        st.session_state.collection = collection
                        st.session_state.embedder = embedder
                        st.session_state.resume_processed = True
                        st.success("✅ Resume processed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.info("Please make sure you've uploaded a valid PDF file with readable text.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Chat section
        st.markdown("### 💬 Chat with Your Resume Assistant")
        
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        
        # Quick question buttons
        st.markdown("### 🎯 Quick Questions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("What are my skills?", use_container_width=True):
                question = "What are my technical skills and competencies?"
                st.session_state.messages.append({"role": "user", "content": question})
                context = get_similar_content(question)
                response = generate_response(question, context)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
            
            if st.button("Work experience", use_container_width=True):
                question = "What is my work experience and employment history?"
                st.session_state.messages.append({"role": "user", "content": question})
                context = get_similar_content(question)
                response = generate_response(question, context)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        
        with col2:
            if st.button("Education background", use_container_width=True):
                question = "What is my educational background and qualifications?"
                st.session_state.messages.append({"role": "user", "content": question})
                context = get_similar_content(question)
                response = generate_response(question, context)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
            
            if st.button("Projects", use_container_width=True):
                question = "What projects have I worked on?"
                st.session_state.messages.append({"role": "user", "content": question})
                context = get_similar_content(question)
                response = generate_response(question, context)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        
        # Chat input
        st.markdown("### 💭 Ask Your Own Question")
        if user_question := st.chat_input("Type your question here..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_question})
            
            # Generate response
            with st.spinner('Searching resume...'):
                context = get_similar_content(user_question)
                response = generate_response(user_question, context)
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            st.rerun()

if __name__ == "__main__":
    main()
