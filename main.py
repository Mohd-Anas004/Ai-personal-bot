import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Personal Assistant", page_icon="🤖", layout="wide")

load_dotenv()

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #0077b5; color: white; }
    .stTextInput>div>div>input { border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZE LLM & MEMORY ---
@st.cache_resource
def load_llm():
    api_key = os.getenv('GroqApi')
    return ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )

llm = load_llm()

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

if 'conversation' not in st.session_state:
    st.session_state.conversation = ConversationChain(
        llm=llm, 
        memory=st.session_state.memory,
        verbose=False
    )

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- SIDEBAR (LinkedIn Branding) ---
with st.sidebar:
    st.title("👨‍💻 Developer Profile")
    st.info("""
    **Mohammad Anas** *Data Science & AI Developer*
    
    📧 mda00400@gmail.com  
    📞 +91 9140495119
    """)
    st.markdown("---")
    st.markdown("### Tech Stack")
    st.code("Python | LangChain\nGroq | Streamlit\n")
    
    if st.button("Clear Conversation"):
        st.session_state.memory.clear()
        st.session_state.chat_history = []
        st.rerun()

# --- MAIN CHAT UI ---
st.title("🤖 AI Personal Assistant")
st.caption("Powered by Llama 3.3 & LangChain | Built for my Personal Use")

# Container for chat history
chat_container = st.container()

# User Input at the bottom
with st.form(key='chat_form', clear_on_submit=True):
    cols = st.columns([8, 2])
    user_input = cols[0].text_input("Ask your assistant...", placeholder="e.g., How can I improve my EDA skills?")
    submit_button = cols[1].form_submit_button(label='Send')

if submit_button and user_input:
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.conversation.predict(input=user_input)
            st.session_state.chat_history.append({"user": user_input, "bot": response})
        except Exception as e:
            st.error(f"Error: {e}")

# Display Chat History
with chat_container:
    for chat in st.session_state.chat_history:
        st.markdown(f"**👤 You:** {chat['user']}")
        st.info(f"**🤖 Assistant:** {chat['bot']}")
        st.markdown("---")