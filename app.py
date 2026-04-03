import streamlit as st
from rag import create_rag_chain
import tempfile
import os

st.title("Hi Aniket \n Chat with your PDF using RAG")

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # ✅ store chain in session state so it doesn't reload every time
    if "chain" not in st.session_state:
        with st.spinner("Loading document..."):
            st.session_state.chain = create_rag_chain(tmp_path)

    # ✅ store chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ✅ show previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # ✅ chat input at bottom
    question = st.chat_input("Ask a question...")

    if question:
        # show user message
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chain.invoke(question)
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})