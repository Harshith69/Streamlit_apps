import streamlit as st
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
import tempfile
import os
import time

# Try to import LangChain components (optional)
try:
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.llms import Ollama
    from langchain.chains import RetrievalQA
    from langchain import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    st.warning("LangChain not available. Using simple similarity search.")

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
    font-size: 3.5rem;
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}

.upload-section {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 3rem;
    border-radius: 20px;
    margin: 2rem 0;
}

.chat-container {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 2rem;
    border-radius: 20px;
    margin: 1rem 0;
}

.user-message {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 1rem;
    border-radius: 15px;
    margin: 0.5rem 0;
    color: white;
}

.assistant-message {
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    padding: 1rem;
    border-radius: 15px;
    margin: 0.5rem 0;
    color: white;
}

.stButton button {
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    color: white;
    border: none;
    padding: 0.7rem 2rem;
    border-radius: 25px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize all session state variables"""
    if 'resume_processed' not in st.session_state:
        st.session_state.resume_processed = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'embedder' not in st.session_state:
        st.session_state.embedder = None

def process_resume_simple(pdf_path):
    """Process resume without LangChain"""
    try:
        # Read PDF using PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        # Split into chunks
        chunks = []
        current_chunk = ""
        for paragraph in text.split('\n\n'):
            if len(current_chunk + paragraph) < 1000:
                current_chunk += paragraph + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Create embeddings
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embedder.encode(chunks).tolist()
        
        # Create ChromaDB collection
        client = chromadb.Client()
        collection = client.create_collection("resume")
        
        # Add to database
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[f"chunk_{i}"]
            )
        
        return collection, embedder
        
    except Exception as e:
        raise Exception(f"Error processing resume: {str(e)}")

def get_ai_response(question):
    """Get AI response using Ollama"""
    try:
        # Get relevant context
        query_embedding = st.session_state.embedder.encode([question]).tolist()
        results = st.session_state.vectorstore.query(
            query_embeddings=query_embedding,
            n_results=3
        )
        
        context = " ".join(results['documents'][0])
        
        # Simple response generation (you can enhance this with Ollama)
        response = f"""Based on the resume information:

{context[:500]}...

This is a simplified response. To get full AI capabilities, make sure Ollama is installed and running with:
```bash
ollama serve
ollama pull llama3.1:8b-instant
```"""
        
        return response
        
    except Exception as e:
        return f"I encountered an error: {str(e)}. Please try again."

def main():
    initialize_session_state()
    
    # Main header
    st.markdown('<div class="main-header">🤖 AI Resume Assistant</div>', unsafe_allow_html=True)
    st.markdown("### Upload your resume and chat with your personal AI assistant!")
    
    # Check if LangChain is available
    if not LANGCHAIN_AVAILABLE:
        st.info("🔧 Using simple mode. Install LangChain for advanced AI features: `pip install langchain`")
    
    # Upload section
    if not st.session_state.resume_processed:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        
        st.markdown("### 📄 Upload Your Resume (PDF only)")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            if st.button("🚀 Process Resume"):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    with st.spinner('Processing your resume...'):
                        collection, embedder = process_resume_simple(tmp_path)
                        st.session_state.vectorstore = collection
                        st.session_state.embedder = embedder
                        st.session_state.resume_processed = True
                        os.unlink(tmp_path)
                    
                    st.success("✅ Resume processed successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat section
    else:
        # Sidebar
        with st.sidebar:
            st.markdown("### 💡 Example Questions")
            examples = [
                "What are my technical skills?",
                "Summarize my work experience",
                "What projects have I worked on?",
                "Tell me about my education"
            ]
            
            for example in examples:
                if st.button(example, use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": example})
                    response = get_ai_response(example)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
            
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        # Chat interface
        st.markdown("### 💬 Chat with Your Resume")
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Ask about your resume..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = get_ai_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

if __name__ == "__main__":
    main()
