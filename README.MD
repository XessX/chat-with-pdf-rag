# 📄 Chat with Your Resume – RAG-Powered PDF Question Answering

[![Live Demo](https://img.shields.io/badge/Try%20It-Live-brightgreen)](https://LIVE_LINK_HERE)

A personal AI assistant that lets you **chat with your resume (or any PDF)** using GPT + LangChain + ChromaDB.

---

## 🔍 Features
- ✅ Upload any PDF
- ✅ Ask questions about its content
- ✅ Powered by Retrieval-Augmented Generation (RAG)
- ✅ Built with Streamlit + OpenAI + LangChain + Chroma

---

## 🧠 How It Works

1. PDF is split into 500-character overlapping chunks  
2. Chunks are embedded using OpenAI embeddings  
3. Stored in a vector DB (Chroma)  
4. GPT (via LangChain) retrieves relevant chunks to answer questions  

---

## 📦 Tech Stack

- Python + Streamlit  
- LangChain (document loader, splitter, retriever, chain)  
- Chroma Vector Store  
- OpenAI (Embeddings + LLM)  
- tiktoken (tokenizer)  

---

## 🚀 Try it Locally

### 1. Clone this repo

```bash
git clone https://github.com/YOUR_USERNAME/chat-with-pdf-rag.git
cd chat-with-pdf-rag

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run it
streamlit run app.py


🔐 You’ll Need an OpenAI API Key
Get one at: https://platform.openai.com/account/api-keys

Paste it in the app when prompted


🧠 Example Prompts
"What is the applicant’s job title?"

"Summarize their experience."

"What programming languages are listed?"

"Which companies has this person worked for?"

🙋‍♂️ Author
Made with ❤️ by Al Jubair Hossain
Feel free to fork, remix, or reach out.