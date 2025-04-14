import streamlit as st
import os
from dotenv import load_dotenv
from utils.pdf_processor import get_pdf_text, get_text_chunks
from utils.llm_utils import get_vector_store, get_conversational_chain

# Load environment variables
load_dotenv()

def init_api():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY not found. Please set it in your .env file.")
        st.stop()
    return api_key

def handle_user_input(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process PDF files first.")
        return

    try:
        response = st.session_state.conversation({
            "question": user_question,
            "chat_history": st.session_state.get("chat_history", [])
        })
        
        st.session_state.chat_history = response['chat_history']

        # Display chat history and source documents
        for i, message in enumerate(st.session_state.chat_history):
            role = "User" if i % 2 == 0 else "Assistant"
            with st.chat_message(role):
                st.markdown(message.content)
        
        # Display source documents if available
        if 'source_documents' in response and response['source_documents']:
            with st.expander("📚 View Source Documents"):
                for i, doc in enumerate(response['source_documents']):
                    st.markdown(f"**Source {i+1}:**")
                    st.markdown(doc.page_content)
                    st.divider()
                
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def main():
    try:
        st.set_page_config("Gen AI", layout="wide")
        st.title("Information Retrieval System")

        # Initialize session state
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Sidebar for PDF upload
        with st.sidebar:
            st.header("📄 Document Upload")
            pdf_docs = st.file_uploader(
                "Upload your PDF files",
                type=['pdf'],
                accept_multiple_files=True
            )
            
            if st.button("Process PDFs", type="primary"):
                if not pdf_docs:
                    st.error("Please upload at least one PDF file.")
                    return

                with st.spinner("Processing PDFs..."):
                    try:
                        api_key = init_api()
                        # Process PDFs
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text.strip():
                            st.error("No text could be extracted from the PDFs.")
                            return
                        
                        # Show processing status
                        status = st.empty()
                        status.text("Splitting text into chunks...")
                        text_chunks = get_text_chunks(raw_text)
                        
                        status.text("Creating vector store...")
                        vector_store = get_vector_store(text_chunks, api_key)
                        
                        status.text("Setting up Q&A chain...")
                        st.session_state.conversation = get_conversational_chain(vector_store, api_key)
                        
                        status.empty()
                        st.success("PDFs processed successfully! You can now ask questions.")
                    except Exception as e:
                        st.error(f"Error processing PDFs: {str(e)}")

        # Main chat interface
        st.divider()
        user_question = st.chat_input("Ask a question about your PDFs...")
        if user_question:
            handle_user_input(user_question)
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
