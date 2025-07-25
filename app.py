import streamlit as st
import os
import time
import shutil
from pathlib import Path
import json
import requests

# Set page config first
st.set_page_config(
    page_title="Bengali RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add rag_code directory to Python path
import sys
rag_code_path = os.path.join(os.path.dirname(__file__), 'rag_code')
if rag_code_path not in sys.path:
    sys.path.append(rag_code_path)

# Import your modules
try:
    from rag_code.db import BengaliVectorStore
    from rag_code.llm import BengaliRAGSystem
    from rag_code.embedder import BengaliEmbedder
    from rag_code.chunker import BengaliChunker
    from rag_code.extractor import extract_text_from_pdf
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.error(f"Python path: {sys.path}")
    st.stop()

# --- Configuration ---
PDF_PATH = os.path.join(os.path.dirname(__file__), "HSC26-Bangla1st-Paper.pdf")
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "bengali_vector_db")
CHAT_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "chat_history.json")

CHUNKER_CONFIG = {
    "max_chunk_size": 400,
    "overlap": 50,
    "min_chunk_size": 200,
}

# --- Custom CSS ---
def load_css():
    st.markdown("""
    <style>
    /* Main container styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 2rem;
        flex-direction: row-reverse;
    }
    
    .chat-message.assistant {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 2rem;
    }
    
    .chat-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        font-weight: bold;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        flex-shrink: 0;
    }
    
    .chat-content {
        flex: 1;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Status indicators */
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    
    /* Progress bar container */
    .progress-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 1rem;
        padding: 1rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        padding: 0.5rem;
        font-size: 0.8rem;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- Utility Functions ---
def save_chat_history(history):
    """Save chat history to file"""
    try:
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Failed to save chat history: {e}")

def load_chat_history():
    """Load chat history from file"""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Failed to load chat history: {e}")
    return []

def clear_database():
    """Clear the vector database"""
    try:
        if os.path.exists(VECTOR_DB_PATH):
            shutil.rmtree(VECTOR_DB_PATH)
        if os.path.exists(CHAT_HISTORY_FILE):
            os.remove(CHAT_HISTORY_FILE)
        return True
    except Exception as e:
        st.error(f"Failed to clear database: {e}")
        return False

def display_chat_message(message, is_user=True):
    """Display a chat message with styling"""
    message_type = "user" if is_user else "assistant"
    avatar = "👤" if is_user else "🤖"
    
    st.markdown(f"""
    <div class="chat-message {message_type}">
        <div class="chat-avatar">{avatar}</div>
        <div class="chat-content">
            <div style="margin-bottom: 0.5rem; font-weight: bold; opacity: 0.8;">
                {'You' if is_user else 'Assistant'}
            </div>
            <div>{message}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system with caching"""
    try:
        embedder = BengaliEmbedder()
        chunker = BengaliChunker(**CHUNKER_CONFIG)
        vector_db = BengaliVectorStore()
        
        if os.path.exists(VECTOR_DB_PATH) and os.path.exists(os.path.join(VECTOR_DB_PATH, "index.pkl")):
            vector_db.load_local(VECTOR_DB_PATH)
            return BengaliRAGSystem(vector_db), None
        else:
            return None, (embedder, chunker, vector_db)
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return None, None

def process_pdf_with_progress():
    """Process PDF with progress bar"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize components
        status_text.text("🔧 Initializing components...")
        progress_bar.progress(10)
        time.sleep(0.5)  # Brief pause for visual feedback
        
        embedder = BengaliEmbedder()
        chunker = BengaliChunker(**CHUNKER_CONFIG)
        vector_db = BengaliVectorStore()
        
        # Extract text
        status_text.text("📄 Extracting text from PDF...")
        progress_bar.progress(20)
        
        text = extract_text_from_pdf(PDF_PATH)
        progress_bar.progress(35)  # Update after extraction
        
        if not text.strip():
            progress_bar.empty()
            status_text.empty()
            st.error("❌ No text could be extracted from the PDF.")
            return None
        
        st.info(f"✅ Extracted {len(text)} characters from PDF")
        
        # Chunk text
        status_text.text("✂️ Chunking text with Bengali chunker...")
        progress_bar.progress(45)
        time.sleep(0.3)
        
        chunks = chunker.chunk_text(text)
        progress_bar.progress(55)
        
        if not chunks:
            progress_bar.empty()
            status_text.empty()
            st.error("❌ No chunks were generated from the text.")
            return None
        
        st.info(f"✅ Created {len(chunks)} chunks")
        
        # Create embeddings and vector store
        status_text.text("🧠 Creating embeddings and vector store...")
        progress_bar.progress(65)
        time.sleep(0.3)
        
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        
        # Extract texts and metadata
        chunk_texts = [chunk["text"] for chunk in chunks]
        chunk_metadata = [chunk["metadata"] for chunk in chunks]
        
        progress_bar.progress(75)
        
        vector_db.create_from_texts(
            texts=chunk_texts,
            metadatas=chunk_metadata
        )
        
        progress_bar.progress(85)
        
        # Save database
        status_text.text("💾 Saving vector database...")
        progress_bar.progress(90)
        time.sleep(0.3)
        
        vector_db.save_local(VECTOR_DB_PATH)
        
        # Initialize RAG system
        status_text.text("🤖 Initializing RAG system...")
        progress_bar.progress(95)
        time.sleep(0.3)
        
        rag_system = BengaliRAGSystem(vector_db)
        
        # Complete
        status_text.text("✅ Processing complete!")
        progress_bar.progress(100)
        
        time.sleep(1.5)  # Show completion
        progress_bar.empty()
        status_text.empty()
        
        return rag_system
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error processing PDF: {e}")
        st.error(f"Error details: {str(e)}")
        return None

# --- Main Application ---
def main():
    load_css()
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; border-radius: 1rem; margin-bottom: 2rem;">
        <h1 style="margin: 0; font-size: 2.5rem;">🤖 Bengali RAG Assistant</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">Enhanced AI Assistant for Bengali & Mixed Language Text</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ System Controls")
        
        # System status
        rag_system, components = initialize_rag_system()
        
        if rag_system:
            st.markdown('<p class="status-success">✅ System Ready</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-warning">⚠️ Database not found</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # PDF Processing
        st.markdown("### 📚 PDF Processing")
        
        if not rag_system:
            if st.button("🚀 Process PDF", type="primary", use_container_width=True):
                # Show processing container
                processing_container = st.empty()
                with processing_container.container():
                    rag_system = process_pdf_with_progress()
                    
                if rag_system:
                    st.success("✅ PDF processed successfully!")
                    # Mark as initialized in session state
                    st.session_state['rag_initialized'] = True
                    # Clear cache to force reload
                    st.cache_resource.clear()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Failed to process PDF. Please check the logs above.")
        else:
            st.success("📖 PDF already processed")
            if st.button("🔄 Reprocess PDF", use_container_width=True):
                # Clear existing data and reprocess
                clear_database()
                st.cache_resource.clear()
                st.rerun()
        
        st.markdown("---")
        
        # Database controls
        st.markdown("### 🗃️ Database Controls")
        
        if st.button("🗑️ Clear Database", type="secondary", use_container_width=True):
            if st.session_state.get('confirm_clear', False):
                if clear_database():
                    st.success("✅ Database cleared!")
                    # Clear the cache and reset session state
                    st.cache_resource.clear()
                    if 'rag_initialized' in st.session_state:
                        del st.session_state['rag_initialized']
                    st.session_state.confirm_clear = False
                    time.sleep(1)
                    st.rerun()
                st.session_state.confirm_clear = False
            else:
                st.session_state.confirm_clear = True
                st.warning("⚠️ Click again to confirm deletion")
        
        if st.button("💬 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            save_chat_history([])
            st.rerun()
        
        st.markdown("---")
        
        # System info
        st.markdown("### 📊 System Info")
        
        if rag_system:
            st.markdown(f"""
            **Chunker Config:**
            - Max size: {CHUNKER_CONFIG['max_chunk_size']}
            - Overlap: {CHUNKER_CONFIG['overlap']}
            - Min size: {CHUNKER_CONFIG['min_chunk_size']}
            """)
        
        st.markdown("---")
        
        # Help
        with st.expander("❓ Help"):
            st.markdown("""
            **How to use:**
            1. Process your PDF first
            2. Ask questions in Bengali or English
            3. View chat history
            4. Clear database when needed
            
            **Features:**
            - Mixed language support
            - Smart discourse markers
            - Context-aware responses
            """)

    # Main chat interface
    if not rag_system:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: rgba(255, 255, 255, 0.1); 
                    border-radius: 1rem; margin: 2rem 0;">
            <h3>🚀 Welcome to Bengali RAG Assistant</h3>
            <p>Please process your PDF file first using the sidebar controls.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = load_chat_history()
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        if st.session_state.chat_history:
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                display_chat_message(question, is_user=True)
                display_chat_message(answer, is_user=False)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; opacity: 0.6;">
                <h4>👋 Start a conversation!</h4>
                <p>Ask me anything about your document in Bengali or English.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask a question...",
                placeholder="আপনার প্রশ্ন লিখুন বা Type your question here...",
                label_visibility="collapsed"
            )
        
        with col2:
            submit = st.form_submit_button("Send", type="primary", use_container_width=True)
    
    # Process user input
    if submit and user_input.strip():
        # Add user message to chat
        display_chat_message(user_input, is_user=True)
        
        # Generate response using API
        with st.spinner("🤔 Thinking..."):
            try:
                # API endpoint configuration
                api_url = "http://127.0.0.1:8000/api/chat/"
                headers = {
                    "Content-Type": "application/json"
                }
                data = {
                    "message": user_input
                }
                
                # Make API request
                api_response = requests.post(api_url, headers=headers, json=data)
                api_response.raise_for_status()  # Raise exception for bad status codes
                
                response = api_response.json()
                answer = response.get('response', 'Sorry, I could not generate a response.')
                
                # Display assistant response
                display_chat_message(answer, is_user=False)
                
                # Update chat history
                st.session_state.chat_history.append((user_input, answer))
                save_chat_history(st.session_state.chat_history)
                
                # Show sources if available
                if response.get('sources'):
                    with st.expander(f"📚 Sources ({len(response['sources'])} chunks)"):
                        for i, source in enumerate(response['sources'][:3], 1):
                            st.markdown(f"""
                            **Source {i}:**
                            - Language: {source.get('language', 'N/A')}
                            - Length: {source.get('length_chars', 'N/A')} chars
                            - Bengali ratio: {source.get('bengali_ratio', 0):.2f}
                            """)
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # Footer
    st.markdown("""
    <div class="footer">
        🤖 Bengali RAG Assistant | Enhanced with Mixed Language Support | Made with ❤️ using Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("❌ GOOGLE_API_KEY environment variable not set!")
        st.info("Please set your Google Generative AI API key in the environment variables.")
        st.code("export GOOGLE_API_KEY='your-api-key-here'")
        st.stop()
    
    main()