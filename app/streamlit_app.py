"""
Streamlit Drug Information Chatbot
Beautiful, interactive web interface for your chatbot

Installation:
pip install streamlit streamlit-chat

Run:
streamlit run streamlit_app.py

Features:
- Chat interface with history
- RAG toggle
- Source citations
- Example questions
- Model info display
"""

import streamlit as st
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

try:
    from notebooks.faiss_rag_builder import DrugRAGRetriever
    RAG_AVAILABLE = True
except Exception as e:
    RAG_AVAILABLE = False
    st.error(f"RAG import error: {e}")

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = ROOT / "models" / "drug_qna_lora" / "final"
FAISS_DIR = ROOT / "data" / "faiss"


# Page config
st.set_page_config(
    page_title="Drug Information Chatbot",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        color: black;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
        color: black;
    }
    .bot-message {
        background-color: #f5f5f5;
        margin-right: 20%;
        color: black;
    }
    .source-box {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        color: black;
    }
    .disclaimer {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
        color: black;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL (with caching)
# ============================================================================

@st.cache_resource
def load_chatbot_model():
    """
    Load model and tokenizer (cached to avoid reloading)
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_DIR,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure the model is trained and saved in the correct directory.")
        return None, None

@st.cache_resource
def load_rag_retriever():
    """
    Load RAG retriever (cached)
    """
    if not RAG_AVAILABLE:
        return None
    
    try:
        retriever = DrugRAGRetriever(FAISS_DIR)
        return retriever
    except Exception as e:
        st.error(f"Error loading RAG: {e}")
        return None

# ============================================================================
# CHAT FUNCTIONS
# ============================================================================

def generate_answer(question, model, tokenizer, retriever=None, use_rag=True):
    """
    Generate answer using model and optional RAG
    """
    context = ""
    sources = []
    
    # Retrieve context if RAG enabled
    if use_rag and retriever:
        try:
            results = retriever.retrieve(question, k=3)
            context = retriever.format_context(results)
            sources = results
        except Exception as e:
            st.warning(f"RAG retrieval failed: {e}")
    
    # Prepare prompt
    if context:
        prompt = f"Context: {context}\n\nQuestion: {question}"
    else:
        prompt = question
    
    # Generate
    try:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
              **inputs, 
              max_length=256,
              num_beams=4,
              repetition_penalty=1.2,
              no_repeat_ngram_size=3,
              early_stopping=True,
            )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer, sources
    
    except Exception as e:
        return f"Error generating answer: {e}", []

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'model' not in st.session_state:
    st.session_state.model, st.session_state.tokenizer = load_chatbot_model()

if 'retriever' not in st.session_state:
    st.session_state.retriever = load_rag_retriever()

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/pharmacy.png", width=100)
    st.title("💊 Drug Info Chatbot")
    
    st.markdown("---")
    
    # Model status
    st.subheader("🤖 System Status")
    if st.session_state.model:
        st.success("✅ Model loaded")
    else:
        st.error("❌ Model not loaded")
    
    if st.session_state.retriever:
        st.success("✅ RAG available")
    else:
        st.warning("⚠️ RAG not available")
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    use_rag = st.checkbox("Enable RAG", value=True, 
                          disabled=not st.session_state.retriever,
                          help="Use retrieval-augmented generation for accurate facts")
    
    temperature = st.slider("Response Creativity", 0.1, 1.0, 0.7, 0.1,
                           help="Higher = more creative, Lower = more focused")
    
    st.markdown("---")
    
    # Example questions
    st.subheader("💡 Try asking:")
    example_questions = [
        "What is the dosage of Ibuprofen?",
        "What are the side effects of Paracetamol?",
        "When should I not take Amoxicillin?",
        "How does Vitamin D work?",
        "Can I take Aspirin with Ibuprofen?"
    ]
    
    for i, question in enumerate(example_questions):
        if st.button(question, key=f"example_{i}"):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()
    
    st.markdown("---")
    
    # Clear chat
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # Model info
    with st.expander("ℹ️ Model Information"):
        st.write(f"**Model:** Flan-T5-Base + LoRA")
        st.write(f"**Training:** Medical QA Dataset")
        st.write(f"**RAG Source:** DailyMed")

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.markdown('<p class="main-header">💊 Drug Information Chatbot</p>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ Important Disclaimer:</strong> This chatbot is for educational purposes only. 
    Always consult with a qualified healthcare professional before making any medical decisions. 
    Do not use this information as a substitute for professional medical advice.
</div>
""", unsafe_allow_html=True)

# Check if model loaded
if not st.session_state.model:
    st.error("❌ Model not loaded. Please train the model first.")
    st.info("Run: `python lora_finetuning.py` to train the model")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    sources = message.get("sources", [])
    
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🧑 You:</strong><br>{content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 Bot:</strong><br>{content}
        </div>
        """, unsafe_allow_html=True)
        
        # Show sources if available
        if sources and use_rag:
            with st.expander("📚 Sources Used"):
                for i, source in enumerate(sources):
                    st.markdown(f"""
                    **Source {i+1}:** {source['drug_name']} - {source['category']}  
                    *Distance: {source['distance']:.4f}*  
                    {source['text'][:200]}...
                    """)

# Chat input
if prompt := st.chat_input("Ask about drug information...", key="chat_input"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.spinner("🤔 Thinking..."):
        answer, sources = generate_answer(
            prompt,
            st.session_state.model,
            st.session_state.tokenizer,
            st.session_state.retriever,
            use_rag
        )
    
    # Add bot message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
    
    # Rerun to display new messages
    st.rerun()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h4>📊 Chat Statistics</h4>
        <p><strong>Messages:</strong> {}</p>
        <p><strong>RAG Status:</strong> {}</p>
    </div>
    """.format(
        len(st.session_state.messages),
        "✅ Enabled" if use_rag else "❌ Disabled"
    ), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h4>🎓 Project Info</h4>
        <p><strong>Course:</strong> NLP/ML Project</p>
        <p><strong>Tech:</strong> Flan-T5 + LoRA + RAG</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h4>🔗 Quick Links</h4>
        <p><a href="https://docs.streamlit.io" target="_blank">Streamlit Docs</a></p>
        <p><a href="https://dailymed.nlm.nih.gov" target="_blank">DailyMed Database</a></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with Streamlit • Powered by Flan-T5 + LoRA + RAG • Educational purposes only")