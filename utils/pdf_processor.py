# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         try:
#             pdf_reader = PdfReader(pdf)
#             for page in pdf_reader.pages:
#                 text += page.extract_text() or ""
#         except Exception as e:
#             st.error(f"Error processing {pdf.name}: {str(e)}")
#     return text

# def get_text_chunks(text):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     return splitter.split_text(text)


import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_pdf_text(pdf_docs):
    try:
        text = ""
        for pdf in pdf_docs:
            try:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    page_text = page.extract_text() or ""
                    if page_text.strip():  # Only add non-empty pages
                        text += page_text + "\n\n"  # Add spacing between pages
            except Exception as e:
                st.error(f"Error processing {pdf.name}: {str(e)}")
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting PDF text: {str(e)}")

def get_text_chunks(text):
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Increased chunk size
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]  # More granular separators
        )
        chunks = splitter.split_text(text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]  # Remove empty chunks
    except Exception as e:
        raise Exception(f"Error splitting text into chunks: {str(e)}")