from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough

# imports stay here

load_dotenv()

def create_rag_chain(file_path):  # ← add this
    model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

    loader = PyMuPDFLoader(file_path = file_path)
    documents = loader.load()  # ← use parameter

    splitter  = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 200)
    splitter_docs = splitter.split_documents(documents=documents)

    embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    vector_stores = FAISS.from_documents(documents = splitter_docs, embedding = embeddings)
    
    #retriever for all the dcuments to load in vector store
    retriever = vector_stores.as_retriever()
    prompt= ChatPromptTemplate.from_messages([
        ("system", "Answer the question based on the context: {context}"),
        ("human","{question}")
    ])

    parser = StrOutputParser()

    chain = ({"context":retriever, "question": RunnablePassthrough()} | prompt | model | parser)

    return chain  # ← add this

# outside function:
#chain = create_rag_chain("Aniket_CV_GenAI.pdf")

#while True:
#    question = input("Quesion: ")
#    if question.lower() == 'exit':
 #       print("chat ended")
  #      break
   # result = chain.invoke(question)
    #print("Answer:", result)
    #print("-" * 50)
    # ... same as before