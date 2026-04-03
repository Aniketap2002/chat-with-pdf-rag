# Chat with PDF using RAG

A RAG based application that lets you upload any PDF 
and ask questions about it using Gemini AI.

## Tech Stack
- LangChain
- Google Gemini
- HuggingFace Embeddings
- FAISS Vector Store
- Streamlit

## How to run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your `GOOGLE_API_KEY` in `.env`
4. Run: `streamlit run app.py`