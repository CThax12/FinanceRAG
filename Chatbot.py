import streamlit as st
from pandas.core.nanops import disallow

from Main import runLLM, GetFAISSVector

st.title("Finance Chatbot using OpenAI Model")

uploadFile = st.file_uploader("Upload an Article", type=("txt", "md", "pdf"))

question = st.text_input(
    "Ask anything about the article",
    placeholder= "Can you summarize the article?",
    disabled= not uploadFile
)

if uploadFile:
    with open(uploadFile.name, "wb") as file:
        file.write(uploadFile.getbuffer())

        vectorDB = GetFAISSVector(uploadFile.name)

        if vectorDB is None:
            st.error(
                f"The {uploadFile.type} is not supported.Please load a file in pdf, txt, or md."
            )
    with st.spinner("Generating response..."):
        if uploadFile and question:
            answer = runLLM(vectorDB=vectorDB, query=question)
            st.write("### Answer")
            st.write(f"{answer}")
