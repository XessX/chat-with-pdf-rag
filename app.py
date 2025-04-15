import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# --- UI START ---
st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF (RAG + ChromaDB)")

# OpenAI API key input
openai_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

# File uploader
uploaded_file = st.file_uploader("📤 Upload a PDF file", type="pdf")

# When both are ready
if openai_key and uploaded_file:
    st.success("✅ File uploaded and API key received. Processing...")

    # Save file locally
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load and split PDF
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    # Create vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_store")

    # Setup Retrieval Chain
    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=openai_key),
        retriever=vectordb.as_retriever()
    )

    # Ask user question
    query = st.text_input("❓ Ask a question about the PDF:")
    if query:
        st.write("🧠 Generating answer...")
        answer = chain.invoke({"query": query})
        st.write("📢 Answer:", answer)
