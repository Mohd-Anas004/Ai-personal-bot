import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory 
from dotenv import load_dotenv
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Anas Intelligence | ▲NIX", 
    page_icon="", 
    layout="wide"
)

load_dotenv()

# ---  DARK THEME ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600&display=swap');

    /* Background: Deep Charcoal Gradient */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #1a1a1a 0%, #0a0a0a 100%);
        color: #ffffff;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Sidebar: Minimalist & Integrated */
    [data-testid="stSidebar"] {
        background-color: #0f0f0f;
        border-right: 1px solid #262626;
    }

    /* Chat Area Layout */
    .chat-wrapper {
        max-width: 850px;
        margin: auto;
        padding-top: 50px;
    }

    /* User Bubble: Vibrant Glassmorphism */
    .user-box {
        align-self: flex-end;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        color: #ffffff;
        padding: 15px 25px;
        border-radius: 24px 24px 4px 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 30px;
        max-width: 80%;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        float: right;
        clear: both;
    }

    /* AI Response: Clean Typography */
    .ai-box {
        align-self: flex-start;
        color: #e0e0e0;
        line-height: 1.8;
        font-size: 1.1rem;
        margin-bottom: 40px;
        padding-left: 10px;
        float: left;
        clear: both;
        width: 100%;
    }

    .ai-icon {
        color: #6366f1;
        font-weight: bold;
        margin-bottom: 5px;
        display: block;
    }

    /* Input Bar: Modern Pill Shape */
    .stChatInputContainer {
        border-radius: 50px !important;
        background-color: #1a1a1a !important;
        border: 1px solid #333333 !important;
        padding: 5px 20px !important;
    }

    /* Loading Spinner Styling */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LLM ENGINE ---
@st.cache_resource
def load_engine():
    api_key = os.getenv('GroqApi')
    if not api_key: return None
    try:
        return ChatGroq(
            api_key=api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.6
        )
    except: return None

llm = load_engine()

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=10)

if 'conversation' not in st.session_state and llm:
    st.session_state.conversation = ConversationChain(
        llm=llm, memory=st.session_state.memory
    )

if 'history' not in st.session_state:
    st.session_state.history = []

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:white;'>▲NIX</h2>", unsafe_allow_html=True)
    st.caption("AI Assistant Engineered by Mohd Anas")
    st.markdown("---")
    st.write("🔧 **Model:** Llama 3.3")
    st.write("⚡ **Provider:** Langchain")
    
    if st.button("New Conversation", use_container_width=True):
        st.session_state.memory.clear()
        st.session_state.history = []
        st.rerun()

# --- CONVERSATION INTERFACE ---
st.markdown("<div class='chat-wrapper'>", unsafe_allow_html=True)

if not st.session_state.history:
    st.markdown("""
        <div style='margin-top:20vh; text-align:center;'>
            <h1 style='font-size:3.5rem; font-weight:600; margin-bottom:0;'>Hello, Anas.</h1>
            <p style='color:#888888; font-size:1.2rem;'>How can I help you be productive today?</p>
        </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.history:
    # User
    st.markdown(f"<div class='user-box'>{msg['user']}</div>", unsafe_allow_html=True)
    # AI
    st.markdown(f"<div class='ai-box'><span class='ai-icon'>▲NIX</span>{msg['bot']}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# --- INPUT ---
prompt = st.chat_input("Ask me anything...")

if prompt:
    if not llm:
        st.error("Missing API Key. Please add it to your secrets.")
    else:
        # Show message immediately
        st.markdown(f"<div class='chat-wrapper'><div class='user-box'>{prompt}</div></div>", unsafe_allow_html=True)
        
        with st.spinner("Processing..."):
            try:
                # Optimized for LangChain 0.3.x
                res = st.session_state.conversation.invoke({"input": prompt})
                st.session_state.history.append({"user": prompt, "bot": res['response']})
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
