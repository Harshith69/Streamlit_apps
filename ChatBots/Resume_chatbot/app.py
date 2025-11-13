import streamlit as st
import pypdf
import tempfile
import os
import re
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import chromadb
from typing import List, Tuple
import numpy as np

def initialize_session_state():
    if 'resume_processed' not in st.session_state:
        st.session_state.resume_processed = False
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = ""
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ""
    if 'first_interaction' not in st.session_state:
        st.session_state.first_interaction = True
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = None
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'chat_memory' not in st.session_state:
        st.session_state.chat_memory = None

@st.cache_resource
def load_embedding_model():
    """Load Hugging Face embedding model"""
    try:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return embeddings
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        return None

def extract_personal_info(text):
    """Better name extraction from resume"""
    name_patterns = [
        r'^([A-Z][a-z]+ [A-Z][a-z]+)',
        r'\n([A-Z][a-z]+ [A-Z][a-z]+)\s*\n',
        r'Name[:]?\s*([A-Z][a-z]+ [A-Z][a-z]+)',
        r'([A-Z][a-z]+ [A-Z][a-z]+)\s+[\w\.-]+@[\w\.-]+',
        r'([A-Z][a-z]+ [A-Z][a-z]+)\s+\+?\d'
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

def extract_text_with_sections(pdf_file):
    """Extract text from PDF with structure preservation"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        with open(tmp_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            text = ""
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"--- PAGE {page_num + 1} ---\n"
                    text += page_text + "\n\n"
        
        os.unlink(tmp_path)
        return text
    except Exception as e:
        raise e

def create_chunks(text, chunk_size=500, chunk_overlap=50):
    """Create text chunks for embedding"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        
        if i + chunk_size >= len(words):
            break
    
    return chunks

def create_vectorstore(text, embeddings):
    """Create ChromaDB vectorstore from resume text"""
    try:
        # Create chunks
        chunks = create_chunks(text)
        
        # Create documents
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={"source": "resume", "chunk_id": i}
            )
            documents.append(doc)
        
        # Create vectorstore
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vectorstore: {str(e)}")
        return None

def setup_retriever(vectorstore):
    """Setup retriever with similarity search"""
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        return retriever
    except Exception as e:
        st.error(f"Error setting up retriever: {str(e)}")
        return None

def create_prompt_template():
    """Create prompt template for the QA system"""
    template = """You are an AI resume assistant representing {name}. Use the following context from the resume to answer the question. 
    If you don't know the answer based on the context, say so. Keep answers professional and relevant to the resume content.

Context: {context}

Conversation History: {history}

Question: {question}

Answer:"""
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "history", "question", "name"]
    )

def format_docs(docs):
    """Format retrieved documents for context"""
    return "\n\n".join([doc.page_content for doc in docs])

def create_qa_chain(retriever, memory, user_name):
    """Create the QA chain with memory and retrieval"""
    prompt = create_prompt_template()
    
    def get_history():
        return memory.load_memory_variables({})["history"]
    
    chain = (
        {
            "context": retriever | format_docs,
            "history": RunnablePassthrough() | (lambda x: get_history()),
            "question": RunnablePassthrough() | (lambda x: x["question"]),
            "name": RunnablePassthrough() | (lambda x: user_name)
        }
        | prompt
        | StrOutputParser()
    )
    
    return chain

def process_resume(pdf_file):
    """Process resume and setup vectorstore"""
    try:
        # Extract text
        text = extract_text_with_sections(pdf_file)
        
        if not text.strip():
            raise ValueError("No readable text found in PDF")
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        st.session_state.resume_text = text
        
        # Extract personal info
        personal_info = extract_personal_info(text)
        st.session_state.user_name = personal_info['name']
        
        # Load embedding model
        with st.spinner("Loading AI model..."):
            embeddings = load_embedding_model()
            if not embeddings:
                raise Exception("Failed to load embedding model")
            
            st.session_state.embedding_model = embeddings
        
        # Create vectorstore
        with st.spinner("Processing resume content..."):
            vectorstore = create_vectorstore(text, embeddings)
            if not vectorstore:
                raise Exception("Failed to create vectorstore")
            
            st.session_state.vectorstore = vectorstore
        
        # Setup retriever
        retriever = setup_retriever(vectorstore)
        if not retriever:
            raise Exception("Failed to setup retriever")
        
        st.session_state.retriever = retriever
        
        # Initialize memory
        st.session_state.chat_memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )
        
        st.success(f"✅ Resume processed! Welcome {st.session_state.user_name}")
        return True
        
    except Exception as e:
        raise e

def generate_ai_response(question):
    """Generate response using the QA chain"""
    try:
        if not st.session_state.retriever or not st.session_state.chat_memory:
            return "System not properly initialized. Please re-upload your resume."
        
        # Create QA chain
        qa_chain = create_qa_chain(
            st.session_state.retriever,
            st.session_state.chat_memory,
            st.session_state.user_name
        )
        
        # Generate response
        response = qa_chain.invoke({"question": question})
        
        # Update memory
        st.session_state.chat_memory.save_context(
            {"input": question},
            {"output": response}
        )
        
        return response
        
    except Exception as e:
        return f"I apologize, but I encountered an issue: {str(e)}. Please try rephrasing your question."

def cleanup_chroma_db():
    """Cleanup ChromaDB files"""
    try:
        if os.path.exists("./chroma_db"):
            import shutil
            shutil.rmtree("./chroma_db")
    except Exception as e:
        print(f"Cleanup warning: {str(e)}")

# UI Code
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
                        # Cleanup previous ChromaDB
                        cleanup_chroma_db()
                        
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
            <p><small>Powered by Hugging Face Embeddings & ChromaDB</small></p>
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
                    response = generate_ai_response(example)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
            
            st.markdown("---")
            if st.button("🔄 Upload New Resume", use_container_width=True):
                # Cleanup
                cleanup_chroma_db()
                
                # Reset session state
                st.session_state.resume_processed = False
                st.session_state.messages = []
                st.session_state.resume_text = ""
                st.session_state.user_name = ""
                st.session_state.first_interaction = True
                st.session_state.vectorstore = None
                st.session_state.retriever = None
                st.session_state.chat_memory = None
                st.rerun()
            
            st.markdown("---")
            st.markdown("### 🔧 System Info")
            st.info("Using: Hugging Face Embeddings + ChromaDB + Conversation Memory")
        
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
            response = generate_ai_response(user_question)
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
                    Powered by Hugging Face + ChromaDB + Streamlit
                </p>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
