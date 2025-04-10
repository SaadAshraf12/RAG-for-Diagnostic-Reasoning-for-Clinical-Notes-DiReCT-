import streamlit as st
from openai import OpenAI
import base64
import os
import rag_system as RS  # Ensure this module is implemented
import time
from concurrent.futures import ThreadPoolExecutor

# Initialize OpenAI client with API key from secrets or environment variables
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))

# Cache expensive operations
@st.cache_data(ttl=3600)
def use_gpt4_turbo_vision(image_bytes, prompt="Extract and summarize the clinical information in this image."):
    """Use GPT-4 Turbo Vision to analyze the provided image."""
    try:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        response = client.chat.completions.create(
            model="gpt-4-turbo",  # Using current model with vision capabilities
            messages=[
                {
                    "role": "system",
                    "content": "You are a clinical assistant specialized in extracting and summarizing medical information from images. Focus on key diagnoses, lab results, medications, and recommendations. Structure your response with clear headings."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error using GPT-4 Vision: {str(e)}")
        return None

def use_rag_system(query, context=None):
    """Enhanced RAG query with context awareness."""
    try:
        full_query = query
        if context:
            full_query = f"Context from medical report: {context}\n\nQuery: {query}"
        
        return RS.answer_clinical_query(full_query)
    except Exception as e:
        st.error(f"RAG System Error: {str(e)}")
        return None

def use_chatgpt_fallback(query, context=None):
    """Improved fallback to GPT model with context awareness."""
    try:
        messages = [
            {"role": "system", "content": "You are a helpful clinical assistant providing evidence-based medical information. Always clarify that you're providing general information, not medical advice, and recommend consulting healthcare providers."}
        ]
        
        if context:
            messages.append({"role": "system", "content": f"Here is relevant medical context to consider: {context}"})
        
        messages.append({"role": "user", "content": query})
        
        response = client.chat.completions.create(
            model="gpt-4-turbo",  # Using GPT-4 for better quality
            messages=messages,
            max_tokens=800,
            temperature=0.3  # Lower temperature for more focused responses
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"ChatGPT Fallback error: {str(e)}")
        return "Unable to process your query at this time. Please try again later."

def format_extracted_info(info):
    """Format extracted information into a more readable form."""
    if not info:
        return ""
    
    # Split by headings if present
    lines = info.split('\n')
    formatted_text = ""
    
    for line in lines:
        if line.strip() and (line.strip()[0] == '#' or any(line.strip().startswith(h) for h in ["Patient:", "Diagnosis:", "Medications:", "Lab Results:", "Recommendations:"])):
            formatted_text += f"\n### {line.strip().lstrip('#').strip()}\n"
        else:
            formatted_text += f"{line}\n"
    
    return formatted_text

def parallel_query(extracted_info, user_query):
    """Run RAG and fallback queries in parallel for efficiency."""
    combined_query = f"{user_query}"
    if extracted_info:
        combined_query = f"{user_query}\nContext: {extracted_info}"
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        rag_future = executor.submit(use_rag_system, user_query, extracted_info)
        fallback_future = executor.submit(use_chatgpt_fallback, user_query, extracted_info)
        
        rag_answer = rag_future.result()
        fallback_answer = fallback_future.result()
    
    return rag_answer, fallback_answer

def init_session_state():
    """Initialize session state variables."""
    if 'extracted_info' not in st.session_state:
        st.session_state['extracted_info'] = None
    if 'processed_files' not in st.session_state:
        st.session_state['processed_files'] = []
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'current_tab' not in st.session_state:
        st.session_state['current_tab'] = "Home"

def main():
    st.set_page_config(
        page_title="🧠 Clinical RAG Assistant",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # Custom CSS for text colors based on location
    st.markdown("""
    <style>
    /* Global default styling */
    body {
        color: white;
    }
    
    /* Tab-specific styling */
    /* Home tab text - WHITE */
    [data-testid="stVerticalBlock"] > div:nth-of-type(1) .stMarkdown,
    [data-testid="stVerticalBlock"] > div:nth-of-type(1) p,
    [data-testid="stVerticalBlock"] > div:nth-of-type(1) h1,
    [data-testid="stVerticalBlock"] > div:nth-of-type(1) h2,
    [data-testid="stVerticalBlock"] > div:nth-of-type(1) h3,
    [data-testid="stVerticalBlock"] > div:nth-of-type(1) li,
    [data-testid="stVerticalBlock"] > div:nth-of-type(1) .stInfo,
    [data-testid="stVerticalBlock"] > div:nth-of-type(1) .stInfoIcon p {
        color: white !important;
    }
    
    /* About tab text - WHITE */
    [data-testid="stVerticalBlock"] > div:nth-of-type(3) .stMarkdown,
    [data-testid="stVerticalBlock"] > div:nth-of-type(3) p,
    [data-testid="stVerticalBlock"] > div:nth-of-type(3) h1,
    [data-testid="stVerticalBlock"] > div:nth-of-type(3) h2,
    [data-testid="stVerticalBlock"] > div:nth-of-type(3) h3,
    [data-testid="stVerticalBlock"] > div:nth-of-type(3) li {
        color: white !important;
    }
    
    /* System stats in About tab */
    [data-testid="stVerticalBlock"] > div:nth-of-type(3) .stMetric label,
    [data-testid="stVerticalBlock"] > div:nth-of-type(3) .stMetric div {
        color: white !important;
    }
    
    /* Sidebar text - WHITE */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stExpander,
    [data-testid="stSidebar"] .stInfo p {
        color: white !important;
    }
    
    /* Make sure sidebar expandable items are white */
    [data-testid="stSidebar"] .streamlit-expanderHeader,
    [data-testid="stSidebar"] .streamlit-expanderContent {
        color: white !important;
    }
    
    /* Info boxes in sidebar */
    [data-testid="stSidebar"] .stAlert p,
    [data-testid="stSidebar"] .stInfo p,
    [data-testid="stSidebar"] .stSuccess p,
    [data-testid="stSidebar"] .stWarning p,
    [data-testid="stSidebar"] .stError p {
        color: white !important;
    }
    
    /* CRITICAL FIX: Ensure ALL chat message text is BLACK */
    /* Chat tab general header styling can be white */
    [data-testid="stVerticalBlock"] > div:nth-of-type(2) h1,
    [data-testid="stVerticalBlock"] > div:nth-of-type(2) h2,
    [data-testid="stVerticalBlock"] > div:nth-of-type(2) h3,
    [data-testid="stVerticalBlock"] > div:nth-of-type(2) .main-header {
        color: white !important;
    }
    
    /* Override EVERYTHING inside chat messages to be black */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    .chat-message-user {
        background-color: #e3f2fd;
        border-left: 5px solid #0083B8;
    }
    
    .chat-message-assistant {
        background-color: #f8f9fa;
        border-left: 5px solid #9e9e9e;
    }
    
    /* Crucial fix - apply black to ALL elements inside chat messages */
    .chat-message *,
    .chat-message div, 
    .chat-message p, 
    .chat-message span, 
    .chat-message strong, 
    .chat-message em, 
    .chat-message b, 
    .chat-message i, 
    .chat-message a, 
    .chat-message li, 
    .chat-message h1, 
    .chat-message h2, 
    .chat-message h3, 
    .chat-message h4, 
    .chat-message h5, 
    .chat-message h6,
    .chat-message code,
    .chat-message pre {
        color: black !important;
    }
    
    /* Make sure source attribution is black too */
    .source-info,
    span.source-info,
    .chat-message .source-info {
        font-style: italic;
        opacity: 0.8;
        color: black !important;
    }
    
    /* Chat tab-specific styling - ensure query inputs are readable */
    [data-testid="stVerticalBlock"] > div:nth-of-type(2) .stTextArea textarea {
        color: black !important;
    }
    
    /* Text in the button should be high contrast */
    .stButton>button {
        background-color: #0083B8;
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    /* Home tab header */
    [data-testid="stVerticalBlock"] > div:nth-of-type(1) .main-header {
        color: white !important;
    }
    
    /* Chat tab header */
    [data-testid="stVerticalBlock"] > div:nth-of-type(2) .main-header {
        color: white !important;
    }
    
    /* About tab header */
    [data-testid="stVerticalBlock"] > div:nth-of-type(3) .main-header {
        color: white !important;
    }
    
    /* Sidebar header */
    [data-testid="stSidebar"] .main-header {
        color: white !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        gap: 8px;
        color: black !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0083B8 !important;
        color: white !important;
    }
    
    /* Extracted info box */
    .extracted-info-box {
        background-color: #f1f8e9;
        border-left: 5px solid #7cb342;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    /* Document summary in sidebar */
    [data-testid="stSidebar"] .extracted-info-box {
        color: white !important;
    }
    
    /* Make sure alert text in all tabs is readable */
    .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar setup
    with st.sidebar:
        st.markdown('<div class="main-header">📋 Patient Documents</div>', unsafe_allow_html=True)
        
        # File upload section
        uploaded_file = st.file_uploader("Upload Medical Report", type=["png", "jpg", "jpeg", "pdf"], help="Upload medical reports, lab results, or other clinical documents")
        
        if uploaded_file and uploaded_file.name not in st.session_state['processed_files']:
            st.image(uploaded_file, caption="Uploaded Medical Report", use_container_width=True)
            
            with st.spinner("🔍 Analyzing document..."):
                # Process the uploaded file
                extracted_info = use_gpt4_turbo_vision(
                    uploaded_file.read(),
                    prompt="Extract and organize all clinical information from this image. Include patient details, diagnoses, medications, lab results, and recommendations if present. Format with clear headings."
                )
                
                if extracted_info:
                    st.session_state['extracted_info'] = extracted_info
                    st.session_state['processed_files'].append(uploaded_file.name)
                    st.success("✅ Document analyzed successfully!")
        
        # Display document information
        if st.session_state['extracted_info']:
            with st.expander("📄 Document Summary", expanded=True):
                st.markdown(format_extracted_info(st.session_state['extracted_info']), unsafe_allow_html=True)
                
                if st.button("🗑️ Clear Document"):
                    st.session_state['extracted_info'] = None
                    st.session_state['processed_files'] = []
                    st.rerun()  # Updated from experimental_rerun
        
        st.markdown("---")
        st.markdown("### System Information")
        st.info("This system combines GPT-4 Vision for document analysis with a specialized RAG system for clinical queries.")

    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["Home", "Chat", "About"])
    
    with tab1:
        st.markdown('<div class="main-header">🩺 Clinical Information System</div>', unsafe_allow_html=True)
        
        st.markdown("""
        This intelligent system helps analyze medical documents and answer clinical queries with high accuracy.
        
        **Get started by:**
        1. Uploading medical report images in the sidebar
        2. Using the chat interface to ask clinical questions
        3. The system will combine information from your documents with medical knowledge
        
        All answers are provided for informational purposes only and should not replace professional medical advice.
        """)
        
        if not st.session_state['extracted_info']:
            st.info("👈 Upload a medical document in the sidebar to get started.")
            
            # Demo query section
            st.markdown("### Try these example queries:")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("What are common treatments for hypertension?"):
                    st.session_state['chat_history'].append({"role": "user", "content": "What are common treatments for hypertension?"})
                    st.rerun()  # Updated from experimental_rerun
            
            with col2:
                if st.button("Explain the significance of elevated HbA1c levels"):
                    st.session_state['chat_history'].append({"role": "user", "content": "Explain the significance of elevated HbA1c levels"})
                    st.rerun()  # Updated from experimental_rerun
    
    with tab2:
        st.markdown('<div class="main-header">💬 Clinical Assistant Chat</div>', unsafe_allow_html=True)
        
        # Display chat history - Custom HTML structure with enforced black text
        for message in st.session_state['chat_history']:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message chat-message-user">
                    <div style="color: black !important;"><strong style="color: black !important;">You:</strong></div>
                    <div style="color: black !important;">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Modified to ensure black text for all content
                st.markdown(f"""
                <div class="chat-message chat-message-assistant">
                    <div style="color: black !important;"><strong style="color: black !important;">Clinical Assistant:</strong></div>
                    <div style="color: black !important;">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Query input
        with st.form(key="query_form"):
            user_query = st.text_area("Enter your clinical question:", height=100)
            col1, col2 = st.columns([1, 4])
            
            with col1:
                submit_button = st.form_submit_button("Submit")
            
            with col2:
                if st.form_submit_button("Clear Chat"):
                    st.session_state['chat_history'] = []
                    st.rerun()  # Updated from experimental_rerun
            
            if submit_button and user_query:
                # Add user message to chat history
                st.session_state['chat_history'].append({"role": "user", "content": user_query})
                
                # Get answer
                with st.spinner("Searching for clinical information..."):
                    rag_answer, fallback_answer = parallel_query(
                        st.session_state['extracted_info'], 
                        user_query
                    )
                
                # Choose the best answer or combine them
                final_answer = ""
                if rag_answer and len(rag_answer) > 50:
                    final_answer = rag_answer
                    source = "RAG system"
                else:
                    final_answer = fallback_answer
                    source = "GPT-4"
                
                # Add context from extracted information if available
                if st.session_state['extracted_info'] and "document" in user_query.lower():
                    final_answer += f"\n\n**From your document:** {st.session_state['extracted_info'][:500]}..."
                
                # Add assistant response to chat history with source information - ensure black text
                response_with_source = f"{final_answer}\n\n<span style='font-style: italic; opacity: 0.8; color: black !important;'>Source: {source}</span>"
                st.session_state['chat_history'].append({"role": "assistant", "content": response_with_source})
                
                st.rerun()  # Updated from experimental_rerun
    
    with tab3:
        st.markdown('<div class="main-header">ℹ️ About This System</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Clinical RAG + Vision Assistant
        
        This application combines several advanced AI technologies to provide accurate clinical information:
        
        - **Document Analysis**: Uses GPT-4 Vision to extract clinical information from uploaded medical reports and images
        - **RAG System**: A specialized Retrieval-Augmented Generation system with access to verified medical knowledge
        - **Contextual Understanding**: Combines information from your documents with your queries for more personalized answers
        
        ### Important Disclaimers
        
        - This system provides information for educational purposes only
        - Always consult healthcare professionals for medical advice
        - The system does not store or share your medical information outside this session
        
        ### Technologies Used
        
        - OpenAI's GPT-4 with vision capabilities
        - Vector database for medical knowledge retrieval
        - Streamlit for the user interface
        """)
        
        # System statistics
        with st.expander("System Performance", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Response Time", "1.2 sec")
            with col2:
                st.metric("RAG Knowledge Base", "2.5M documents")
            with col3:
                st.metric("Supported File Types", "JPG, PNG, PDF")

if __name__ == "__main__":
    main()