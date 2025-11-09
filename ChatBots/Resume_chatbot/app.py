import streamlit as st
import pypdf
import tempfile
import os
import re
import requests
import json
from datetime import datetime

def initialize_session_state():
    if 'resume_processed' not in st.session_state:
        st.session_state.resume_processed = False
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = ""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'first_interaction' not in st.session_state:
        st.session_state.first_interaction = True

def extract_personal_info(text):
    """Extract name and basic info from resume"""
    # Extract name (look for patterns like "John Doe" at beginning)
    name_pattern = r'^([A-Z][a-z]+ [A-Z][a-z]+)'
    name_match = re.search(name_pattern, text)
    name = name_match.group(1) if name_match else "the candidate"
    
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, text)
    email = email_match.group() if email_match else ""
    
    # Extract skills
    skills_keywords = ['python', 'java', 'javascript', 'sql', 'aws', 'docker', 'react', 'node', 'machine learning', 'data analysis']
    found_skills = [skill for skill in skills_keywords if skill in text.lower()]
    
    return {
        'name': name,
        'email': email,
        'skills': found_skills[:5]
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
        
        os.unlink(tmp_path)
        return True
        
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise e

def get_conversation_context():
    """Get recent conversation history for context"""
    if len(st.session_state.messages) > 0:
        recent_chat = st.session_state.messages[-6:]  # Last 3 exchanges
        context = "\nRecent conversation:\n"
        for msg in recent_chat:
            role = "Human" if msg["role"] == "user" else "AI"
            context += f"{role}: {msg['content']}\n"
        return context
    return ""

def generate_intelligent_response(question, resume_text):
    """Generate intelligent response with memory and context"""
    
    # Get conversation context
    conversation_context = get_conversation_context()
    
    # Extract relevant information from resume
    relevant_context = extract_relevant_info(question, resume_text)
    
    # Create comprehensive prompt
    prompt = f"""
    You are {st.session_state.user_name}, a professional candidate. You are having a conversation about your resume and experience.
    
    RESUME INFORMATION:
    {relevant_context}
    
    {conversation_context}
    
    CURRENT QUESTION: {question}
    
    PERSONALITY & GUIDELINES:
    1. You ARE {st.session_state.user_name} - speak in first person as if you are the person from the resume
    2. Be professional, helpful, and engaging
    3. If information is not in the resume, politely apologize and suggest what you CAN discuss
    4. For career development questions, provide constructive advice based on your experience
    5. Maintain conversation flow and remember previous context
    6. Be specific about your skills and experiences
    7. If asked for advice, draw from your professional journey
    
    RESPONSE STRUCTURE:
    - Acknowledge the question naturally
    - Provide specific information from your resume
    - Offer additional insights or advice if relevant
    - Keep it conversational but professional
    
    YOUR RESPONSE:
    """
    
    # For now, using smart formatting - you can replace with actual AI model
    return format_ai_response(question, relevant_context, conversation_context)

def extract_relevant_info(question, resume_text):
    """Extract highly relevant information based on question"""
    question_lower = question.lower()
    resume_lower = resume_text.lower()
    
    # Define question categories and their keywords
    categories = {
        'skills': ['skill', 'technology', 'programming', 'framework', 'tool', 'language', 'technical', 'proficient', 'expert'],
        'experience': ['experience', 'work', 'job', 'employment', 'role', 'position', 'company', 'worked at', 'responsibilities'],
        'education': ['education', 'degree', 'university', 'college', 'course', 'qualification', 'graduated', 'academic'],
        'projects': ['project', 'portfolio', 'built', 'created', 'developed', 'implemented', 'led', 'managed'],
        'achievements': ['achievement', 'award', 'recognition', 'accomplishment', 'success', 'result'],
        'career_advice': ['advice', 'suggest', 'improve', 'develop', 'growth', 'career', 'future', 'next steps', 'how to']
    }
    
    # Score sentences based on relevance
    sentences = re.split(r'[.!?]+', resume_text)
    scored_sentences = []
    
    for sentence in sentences:
        if len(sentence.strip()) < 20:
            continue
            
        sentence_lower = sentence.lower()
        score = 0
        
        # Basic keyword matching
        question_words = set(question_lower.split())
        sentence_words = set(sentence_lower.split())
        common_words = question_words.intersection(sentence_words)
        score += len(common_words) * 3
        
        # Category-based scoring
        for category, keywords in categories.items():
            if any(keyword in question_lower for keyword in keywords):
                if any(keyword in sentence_lower for keyword in keywords):
                    score += 5
        
        if score > 0:
            scored_sentences.append((score, sentence.strip()))
    
    # Get top relevant sentences
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    relevant_sentences = [sentence for score, sentence in scored_sentences[:7]]
    
    return ". ".join(relevant_sentences) if relevant_sentences else "I don't have specific information about this in my resume."

def format_ai_response(question, context, conversation_context):
    """Format intelligent response based on context and question type"""
    
    question_lower = question.lower()
    
    # Greeting for first interaction
    if st.session_state.first_interaction:
        st.session_state.first_interaction = False
        return f"Hello! I'm {st.session_state.user_name}. Thank you for reviewing my resume! I'd be happy to discuss my experience, skills, and background. How can I assist you in understanding my qualifications better?"
    
    # Check if context is available
    if "don't have specific information" in context.lower():
        return f"I apologize, but I don't have specific information about {question} in my resume. However, I'd be happy to discuss my skills, work experience, education, or projects that are detailed in my resume. What would you like to know about my background?"
    
    # Career advice and development questions
    if any(word in question_lower for word in ['advice', 'suggest', 'improve', 'develop', 'growth', 'career', 'how to']):
        return provide_career_advice(question, context)
    
    # Specific question types with tailored responses
    if any(word in question_lower for word in ['skill', 'technology', 'programming']):
        return f"Based on my experience, here are my key technical skills:\n\n{context}\n\nI'm always looking to expand my skill set. Are there any specific technologies you'd like to know more about?"
    
    elif any(word in question_lower for word in ['experience', 'work', 'job']):
        return f"Regarding my professional experience:\n\n{context}\n\nI've gained valuable insights through these roles that I'd be happy to discuss further."
    
    elif any(word in question_lower for word in ['education', 'degree', 'university']):
        return f"About my educational background:\n\n{context}\n\nMy education has provided me with a strong foundation that I've built upon throughout my career."
    
    elif any(word in question_lower for word in ['project', 'portfolio']):
        return f"Here are some projects I've worked on:\n\n{context}\n\nThese projects have helped me develop both technical and collaborative skills."
    
    else:
        return f"Regarding your question about {question}:\n\n{context}\n\nIs there anything specific about this you'd like me to elaborate on?"

def provide_career_advice(question, context):
    """Provide career development advice"""
    advice_topics = {
        'technical': [
            "Consider working on personal projects to apply new technologies",
            "Contribute to open-source projects to gain collaborative experience",
            "Stay updated with industry trends through blogs and online courses",
            "Practice coding challenges to improve problem-solving skills"
        ],
        'softskills': [
            "Develop communication skills through presentations and documentation",
            "Seek mentorship opportunities for professional growth",
            "Practice giving and receiving constructive feedback",
            "Work on time management and project planning skills"
        ],
        'career': [
            "Set clear short-term and long-term career goals",
            "Network with professionals in your desired field",
            "Consider certifications relevant to your career path",
            "Seek diverse experiences to build a well-rounded profile"
        ]
    }
    
    if 'technical' in question.lower() or 'coding' in question.lower() or 'programming' in question.lower():
        advice = advice_topics['technical']
    elif 'soft' in question.lower() or 'communication' in question.lower():
        advice = advice_topics['softskills']
    else:
        advice = advice_topics['career']
    
    selected_advice = advice[:2]  # Take 2 most relevant advice points
    
    return f"Based on my professional journey, here's some advice that might be helpful:\n\n" + "\n".join([f"• {point}" for point in selected_advice]) + f"\n\nFrom my experience: {context}"

def simple_search(question, text):
    return generate_intelligent_response(question, text)

def generate_response(question, context):
    return generate_intelligent_response(question, context)

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
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
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
    border: 1px solid rgba(255,255,255,0.1);
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
        st.markdown("### Upload your resume and chat with an AI version of yourself!")
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        st.markdown("### 📄 Upload Your Resume (PDF only)")
        
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed")
        
        if uploaded_file is not None:
            st.info(f"📁 File: {uploaded_file.name}")
            
            if st.button("🚀 Process Resume", type="primary"):
                with st.spinner('Processing your resume and creating your AI assistant...'):
                    try:
                        if process_resume(uploaded_file):
                            st.session_state.resume_processed = True
                            st.success("✅ Resume processed successfully! Your AI assistant is ready.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Welcome banner with personal info
        st.markdown(f"""
        <div class="welcome-banner">
            <h3>👋 Hello! I'm {st.session_state.user_name}</h3>
            <p>I'm here to discuss my experience, skills, and background. Feel free to ask me anything about my resume!</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.sidebar:
            st.markdown("### 💡 Conversation Starters")
            examples = [
                "Can you introduce yourself?",
                "What are your strongest technical skills?",
                "Tell me about your work experience",
                "What projects are you most proud of?",
                "How do you stay updated with technology?",
                "What career advice would you give?",
                "What are your professional goals?"
            ]
            
            for example in examples:
                if st.button(example, use_container_width=True):
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
                st.rerun()
        
        # Display chat messages
        st.markdown("### 💬 Conversation")
        
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
        if user_question := st.chat_input(f"Ask {st.session_state.user_name} about their experience..."):
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
                    Powered by AI & Streamlit | Intelligent Resume Assistant with Memory
                </p>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
