import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

# ─────────────────────────────────────────────────────────────
# 🔐 Load environment variable
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 🖼 UI Setup
st.set_page_config(page_title="Chat with your Resume", page_icon="📄")
st.title("📄 Chat with your Resume (RAG + GPT)")

st.markdown("""
Upload your **PDF resume**, and ask questions about it using GPT-powered RAG.
""")

# 📤 PDF Upload
uploaded_file = st.file_uploader("📤 Upload your resume (PDF only)", type="pdf")

# 🔧 Chunk Configuration
st.subheader("🔧 Chunking Settings")
chunk_size = st.slider("🧩 Chunk Size", min_value=100, max_value=2000, value=500, step=100)
chunk_overlap = st.slider("🔁 Chunk Overlap", min_value=0, max_value=500, value=50, step=25)

# 🧠 Process if all required inputs are available
if uploaded_file and openai_key:
    try:
        st.success("✅ Resume uploaded. Processing...")

        # Save the uploaded file temporarily
        temp_path = "temp_resume.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load PDF and chunk
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(docs)

        # Embed chunks & initialize Chroma
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)

        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(openai_api_key=openai_key),
            retriever=retriever,
            return_source_documents=True
        )

        # 💬 Ask question
        query = st.text_input("💬 Ask a question about your resume")

        if query:
            with st.spinner("🧠 Thinking..."):
                result = chain.invoke({"query": query})
                st.markdown("### 📢 Answer")
                st.success(result.get("result", "No answer found."))

                with st.expander("🧩 Source Chunks Used"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.markdown(f"**Chunk {i+1}:**\n```text\n{doc.page_content}\n```")

        # 🧹 Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    except Exception as e:
        st.error(f"❌ Error: {e}")

# 🔐 API key missing
elif uploaded_file and not openai_key:
    st.warning("🔐 OpenAI API key not found. Please set it in your `.env` file as `OPENAI_API_KEY=...`.")
