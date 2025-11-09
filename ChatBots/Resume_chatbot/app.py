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
    if 'resume_chunks' not in st.session_state: 
        st.session_state.resume_chunks = []  
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ""
    if 'first_interaction' not in st.session_state:
        st.session_state.first_interaction = True
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None

@st.cache_resource
def load_ai_model():
    """API-based AI fallback - no local model needed"""
    return None

def extract_personal_info(text):
    """Better name extraction from resume"""
    # Multiple patterns to catch name in different formats
    name_patterns = [
        r'^([A-Z][a-z]+ [A-Z][a-z]+)',
        r'\n([A-Z][a-z]+ [A-Z][a-z]+)\s*\n',
        r'Name[:]?\s*([A-Z][a-z]+ [A-Z][a-z]+)',
        r'([A-Z][a-z]+ [A-Z][a-z]+)\s+[\w\.-]+@[\w\.-]+',  # Name followed by email
        r'([A-Z][a-z]+ [A-Z][a-z]+)\s+\+?\d'  # Name followed by phone
    ]
    
    name = "the candidate"
    for pattern in name_patterns:
        match = re.search(pattern, text)
        if match:
            name = match.group(1)
            break
    
    # If still not found, take first line that looks like a name
    if name == "the candidate":
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+$', line):
                name = line
                break
    
    return {'name': name}

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
        
        # Load AI model
        st.session_state.chatbot = None
        
        os.unlink(tmp_path)
        return True
        
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise e

def extract_text_with_sections(pdf_file):
    """Better text extraction that preserves sections and structure"""
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
                    # Add page separator to maintain structure
                    text += f"--- PAGE {len(text.split('--- PAGE'))} ---\n"
                    text += page_text + "\n\n"
        
        os.unlink(tmp_path)
        return text
    except Exception as e:
        raise e

def smart_text_chunking(text):
    """Better chunking that preserves context and sections"""
    # First, try to split by sections (common resume headings)
    sections = {
        'experience': [],
        'education': [], 
        'skills': [],
        'projects': [],
        'summary': [],
        'other': []
    }
    
    # Common resume section headers
    section_headers = {
        'experience': ['experience', 'work experience', 'employment', 'professional experience', 'work history'],
        'education': ['education', 'academic', 'qualifications', 'degrees'],
        'skills': ['skills', 'technical skills', 'competencies', 'technologies'],
        'projects': ['projects', 'personal projects', 'portfolio', 'key projects'],
        'summary': ['summary', 'objective', 'about', 'profile']
    }
    
    lines = text.split('\n')
    current_section = 'other'
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Check if this line is a section header
        for section, headers in section_headers.items():
            if any(header in line_lower for header in headers) and len(line) < 100:
                current_section = section
                continue
        
        # Add content to appropriate section
        if line.strip() and len(line.strip()) > 10:  # Meaningful content
            sections[current_section].append(line.strip())
    
    # Create chunks from sections
    chunks = []
    
    # Experience chunks
    if sections['experience']:
        exp_text = ' '.join(sections['experience'])
        exp_chunks = [exp_text[i:i+800] for i in range(0, len(exp_text), 800)]
        chunks.extend([f"WORK EXPERIENCE: {chunk}" for chunk in exp_chunks])
    
    # Education chunks
    if sections['education']:
        edu_text = ' '.join(sections['education'])
        edu_chunks = [edu_text[i:i+600] for i in range(0, len(edu_text), 600)]
        chunks.extend([f"EDUCATION: {chunk}" for chunk in edu_chunks])
    
    # Skills chunks
    if sections['skills']:
        skills_text = ' '.join(sections['skills'])
        skills_chunks = [skills_text[i:i+500] for i in range(0, len(skills_text), 500)]
        chunks.extend([f"SKILLS: {chunk}" for chunk in skills_chunks])
    
    # Projects chunks
    if sections['projects']:
        proj_text = ' '.join(sections['projects'])
        proj_chunks = [proj_text[i:i+700] for i in range(0, len(proj_text), 700)]
        chunks.extend([f"PROJECTS: {chunk}" for chunk in proj_chunks])
    
    # If no sections detected, fall back to sentence-based chunking
    if not chunks:
        sentences = re.split(r'[.!?]+', text)
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk + sentence) < 800:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    return chunks

def process_resume(pdf_file):
    try:
        # Extract text with better structure
        text = extract_text_with_sections(pdf_file)
        
        if not text.strip():
            raise ValueError("No readable text found in PDF")
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Smart chunking
        chunks = smart_text_chunking(text)
        
        # Store both full text and chunks
        st.session_state.resume_text = text
        st.session_state.resume_chunks = chunks
        
        # Extract personal info
        personal_info = extract_personal_info(text)
        st.session_state.user_name = personal_info['name']
        st.session_state.personal_info = personal_info
        
        st.success(f"✅ Resume processed! Found {len(chunks)} information sections")
        return True
        
    except Exception as e:
        raise e

def extract_relevant_context(question, resume_text):
    """Much better context extraction using both full text and chunks"""
    try:
        question_lower = question.lower()
        
        # Use both full text and chunks for better matching
        all_text = resume_text
        if hasattr(st.session_state, 'resume_chunks'):
            all_text += " " + " ".join(st.session_state.resume_chunks)
        
        # Better keyword matching with section awareness
        sections_to_search = []
        
        if any(word in question_lower for word in ['skill', 'technology', 'programming', 'technical']):
            sections_to_search.extend(['SKILLS:', 'TECHNICAL:', 'COMPETENCIES:'])
        
        if any(word in question_lower for word in ['experience', 'work', 'job', 'employment']):
            sections_to_search.extend(['EXPERIENCE:', 'WORK:', 'EMPLOYMENT:'])
        
        if any(word in question_lower for word in ['education', 'degree', 'university', 'college']):
            sections_to_search.extend(['EDUCATION:', 'ACADEMIC:'])
        
        if any(word in question_lower for word in ['project', 'portfolio']):
            sections_to_search.extend(['PROJECTS:', 'PORTFOLIO:'])
        
        # Search in relevant sections first
        relevant_content = []
        
        # If we have chunks with sections, search there first
        if hasattr(st.session_state, 'resume_chunks'):
            for chunk in st.session_state.resume_chunks:
                for section in sections_to_search:
                    if section in chunk:
                        relevant_content.append(chunk)
                        break
        
        # If no section matches found, use keyword matching on full text
        if not relevant_content:
            sentences = re.split(r'[.!?]+', all_text)
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
                score += len(common_words) * 5
                
                # Bonus for section keywords
                for section in sections_to_search:
                    if section.lower() in sentence_lower:
                        score += 10
                
                if score > 0:
                    scored_sentences.append((score, sentence.strip()))
            
            scored_sentences.sort(reverse=True, key=lambda x: x[0])
            relevant_content = [sentence for score, sentence in scored_sentences[:5]]
        
        # If still no content, return a helpful message
        if not relevant_content:
            return f"No specific information found about '{question}'. Please try asking about: skills, work experience, education, or projects."
        
        return ". ".join(relevant_content) if isinstance(relevant_content, list) else relevant_content
        
    except Exception as e:
        return f"Error processing question: {str(e)}"

def get_conversation_history():
    """Get conversation context"""
    if len(st.session_state.messages) == 0:
        return ""
    
    history = "\nPrevious conversation:\n"
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
    relevant_sentences = [sentence for score, sentence in scored_sentences[:4]]
    
    return ". ".join(relevant_sentences) if relevant_sentences else "No specific information available."

def generate_ai_response(question, resume_text):
    """Generate response using API-based AI (currently using smart fallback)"""
    try:
        # For now, using enhanced smart response system
        # You can later integrate with Hugging Face Inference API or other services
        context = extract_relevant_context(question, resume_text)
        conversation_history = get_conversation_history()
        
        # Enhanced smart response based on question type and context
        question_lower = question.lower()
        
        if "no specific information" in context.lower():
            return f"I apologize, but I don't have specific information about '{question}' in my resume. I'd be happy to discuss my skills, work experience, education background, or projects I've worked on instead."
        
        # Career advice questions
        if any(word in question_lower for word in ['advice', 'suggest', 'improve', 'how to', 'career']):
            advice = ""
            if any(word in question_lower for word in ['technical', 'coding', 'programming']):
                advice = "For technical growth, I recommend working on real-world projects, contributing to open source, and staying updated with latest technologies through online courses and communities."
            elif any(word in question_lower for word in ['soft skill', 'communication', 'teamwork']):
                advice = "For soft skills, I suggest practicing public speaking, seeking mentorship, participating in team projects, and actively seeking feedback to improve."
            else:
                advice = "Based on my experience, continuous learning, networking, and taking on challenging projects are key for career growth."
            
            return f"Regarding your question about {question}:\n\n{context}\n\nCareer Advice: {advice}"
        
        # Introduction/greeting
        elif any(word in question_lower for word in ['hello', 'hi', 'hey', 'introduce', 'who are you']):
            return f"Hello! I'm {st.session_state.user_name}. Thank you for your interest in my profile! {context}\n\nI'd be happy to discuss my experience in more detail. What would you like to know about my background?"
        
        # Skills questions
        elif any(word in question_lower for word in ['skill', 'technology', 'programming', 'technical']):
            return f"Based on my resume, here are my key technical skills and competencies:\n\n{context}\n\nI'm always expanding my skill set through continuous learning and practical application."
        
        # Experience questions
        elif any(word in question_lower for word in ['experience', 'work', 'job', 'employment']):
            return f"Regarding my professional experience:\n\n{context}\n\nThis experience has provided me with valuable insights and capabilities that I bring to every project."
        
        # Education questions
        elif any(word in question_lower for word in ['education', 'degree', 'university', 'college']):
            return f"About my educational background:\n\n{context}\n\nMy education has given me a strong foundation that I've built upon throughout my career."
        
        # Project questions
        elif any(word in question_lower for word in ['project', 'portfolio', 'work on']):
            return f"Here are some projects I've worked on:\n\n{context}\n\nThese projects have helped me develop both technical expertise and collaborative skills."
        
        # General questions
        else:
            return f"Regarding your question about {question}:\n\n{context}\n\nIs there anything specific about this you'd like me to elaborate on?"
            
    except Exception as e:
        return f"I apologize, but I encountered an issue while processing your question. Please try asking about my skills, experience, education, or projects."

def generate_smart_response(question, resume_text):
    """Smart rule-based response as fallback"""
    context = extract_relevant_context(question, resume_text)
    
    if "no specific information" in context.lower():
        return f"I don't have specific information about '{question}' in my resume. I'd be happy to discuss my skills, experience, education, or projects."
    
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['skill', 'technology', 'programming']):
        return f"Based on my resume, here are my relevant skills:\n\n{context}"
    
    elif any(word in question_lower for word in ['experience', 'work', 'job']):
        return f"Regarding my professional experience:\n\n{context}"
    
    elif any(word in question_lower for word in ['education', 'degree', 'university']):
        return f"About my educational background:\n\n{context}"
    
    elif any(word in question_lower for word in ['project', 'portfolio']):
        return f"Here are some projects from my experience:\n\n{context}"
    
    elif any(word in question_lower for word in ['advice', 'suggest', 'improve']):
        return f"Based on my journey: {context}\n\nI'd suggest continuous learning and practical application."
    
    else:
        return f"Regarding your question:\n\n{context}"

def generate_response(question, resume_text):
    """Try AI first, then fallback"""
    ai_response = generate_ai_response(question, resume_text)
    if ai_response:
        return ai_response
    return generate_smart_response(question, resume_text)

# UI Code (same as before)
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
            <p>AI-Powered Resume Assistant | Ready to discuss my experience</p>
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
                    Powered by AI & Streamlit | Intelligent Resume Assistant
                </p>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
