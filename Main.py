import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader

def GetFAISSVector(file: str):
    fileName, fileExtension = os.path.splitext(file)
    embedding = OpenAIEmbeddings()

    faissIndexPath = f"faiss_index_{fileName}"

    if fileExtension == ".pdf":
        loader = PyPDFLoader(file_path=file)
    elif fileExtension == ".txt":
        loader = TextLoader(file_path=file)
    elif fileExtension == ".md":
        loader = UnstructuredMarkdownLoader(file_path=file)
    else:
        print("This document type is not supported.")
        return None

    documents = loader.load()
    textSplitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=30,
        separators=["\n", "\n\n", "(?<=/. )", ", ", " "]
    )

    chunkedDocs = textSplitter.split_documents(documents=documents)
    vectorDB = FAISS.from_documents(chunkedDocs, embedding)
    vectorDB.save_local(faissIndexPath)

    return vectorDB