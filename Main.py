import os

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

def runLLM(vectorDB, query: str) -> str:
    openAILLM = ChatOpenAI(temperature=0, verbose=True)
    retrievalQA = RetrievalQA.from_chain_type(
        llm=openAILLM, chain_type="stuff", retriever=vectorDB.as_retriever()
    )

    answer = retrievalQA.run(query)

    return answer

