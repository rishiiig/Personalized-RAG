from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.llm import LLMChain

def get_vector_store(text_chunks, api_key):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        return FAISS.from_texts(text_chunks, embeddings)
    except Exception as e:
        raise Exception(f"Error creating vector store: {str(e)}")

def get_conversational_chain(vector_store, api_key):
    try:
        llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model="gemini-pro",
            temperature=0.3
        )
        
        # Custom prompt template for the question answering
        qa_prompt_template = """
        Answer the question based on the context provided.
        If you cannot find the answer in the context, Just give the answer from the most relevant part of the source which you find, but it should be very reliable answer about.
        Even after that, if you even cannot find the answer/relevant answer in the context, tell the exact reason why you cannot find the answer and keep the reason very short and simple.
        The user is a developer so you can give the exact technical reason so that it helps the developer to fix the issue.

        Context: {context}
        Question: {question}
        
        Answer the question in a detailed and helpful way. If possible, cite specific information from the context.
        
        Answer:"""

        # Custom prompt template for question generation
        condense_prompt_template = """
        Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
        
        QA_PROMPT = PromptTemplate(
            template=qa_prompt_template,
            input_variables=["context", "question"]
        )
        
        CONDENSE_QUESTION_PROMPT = PromptTemplate(
            template=condense_prompt_template,
            input_variables=["chat_history", "question"]
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )

        question_generator = LLMChain(
            llm=llm, 
            prompt=CONDENSE_QUESTION_PROMPT
        )

        qa_chain = load_qa_chain(
            llm=llm,
            chain_type="stuff",
            prompt=QA_PROMPT
        )
        
        return ConversationalRetrievalChain(
            retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
            memory=memory,
            combine_docs_chain=qa_chain,
            question_generator=question_generator,
            return_source_documents=True,
            rephrase_question=True
        )
    except Exception as e:
        raise Exception(f"Error creating conversational chain: {str(e)}")