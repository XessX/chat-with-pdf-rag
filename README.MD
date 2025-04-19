# 📄 Chat with Your Resume – RAG-Powered PDF Question Answering

[![Live Demo](https://img.shields.io/badge/Try%20It-Live-brightgreen)](https://chat-with-pdf-rag.onrender.com)

Chat with your resume (or any PDF) using GPT + LangChain + ChromaDB.  
This app breaks your PDF into chunks, stores them in a vector database, and lets GPT answer your questions via RAG (Retrieval-Augmented Generation).

---

## 🔍 Features

- 📤 Upload any PDF resume (or document)
- 💬 Ask natural language questions
- 🧠 Uses GPT + RAG to give intelligent answers
- 🔗 Embeds + indexes using OpenAI + ChromaDB
- 🖼 Simple Streamlit-based UI

---

## 🧠 How It Works

1. Your PDF is split into ~500-character chunks  
2. Text chunks are embedded via OpenAI Embeddings  
3. Chunks are indexed into Chroma (vector DB)  
4. GPT retrieves and answers based on relevant content  

---

## 📦 Tech Stack

- **Python** + **Streamlit**  
- **LangChain** (PDF loader, splitter, RAG chain)  
- **OpenAI** (Embeddings + LLM)  
- **Chroma** (Vector DB)  
- `tiktoken` for token-based splitting  

---

## 🚀 Getting Started (Local)

### 1. Clone this Repo

```bash
git clone https://github.com/XessX/chat-with-pdf-rag.git
cd chat-with-pdf-rag

2. Create a .env File
env

OPENAI_API_KEY=your-key-here

3. Install Requirements

pip install -r requirements.txt

4. Run the App

streamlit run app.py

🧠 Example Prompts

"What is the applicant’s job title?"
"Summarize their experience."
"Which programming languages are listed?"
"Which companies has this person worked for?"

🛠 Project Structure

chat-with-pdf-rag/
├── app.py                # Main Streamlit App
├── .env.example          # Sample env file
├── requirements.txt      # Dependencies
├── README.md             # This file
🔐 Note on OpenAI Key
You can get yours here:
👉 https://platform.openai.com/account/api-keys

Add it to .env or input it at runtime.

🙋‍♂️ Author
Made with ❤️ by Al Jubair Hossain

📎 GitHub: @XessX

📎 LinkedIn: al-jubair-hossain

🧠 Have fun chatting with your PDFs!