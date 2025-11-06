import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "")


st.set_page_config(
    page_title="Gemma Chatbot 💬",
    page_icon="🤖",
    layout="centered",
)

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #f9fafc;
        }
        .stTextInput>div>div>input {
            border: 1px solid #d3d3d3;
            border-radius: 10px;
            padding: 8px;
        }
        .chat-bubble {
            background-color: #e6f0ff;
            padding: 12px 18px;
            border-radius: 12px;
            margin-bottom: 10px;
            width: fit-content;
            max-width: 85%;
        }
        .bot-bubble {
            background-color: #dcf8c6;
            padding: 12px 18px;
            border-radius: 12px;
            margin-bottom: 10px;
            width: fit-content;
            max-width: 85%;
        }
    </style>
""", unsafe_allow_html=True)


st.title("🤖 LangChain Chatbot with Gemma")
st.markdown("Ask anything and let the Gemma model respond intelligently!")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the question asked."),
    ("user", "Question: {question}")
])

llm = Ollama(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser


st.write("---")
user_question = st.text_input("💭 What question do you have in mind?", placeholder="Type your question here...")


if user_question:
    with st.spinner("Thinking... 🤔"):
        response = chain.invoke({"question": user_question})

    st.write("### 🧠 Response:")
    st.markdown(f"<div class='bot-bubble'>{response}</div>", unsafe_allow_html=True)


st.markdown("---")
st.caption("Built with ❤️ using Streamlit, LangChain, and Gemma (via Ollama).")
