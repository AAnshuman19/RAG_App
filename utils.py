# utils.py
import os
import glob
from typing import List
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader, JSONLoader, TextLoader
DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_DIR = "data"

os.makedirs(DB_FAISS_PATH, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def save_uploaded_files(uploaded_files) -> List[str]:
    saved_files = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(file_path)
    return saved_files

def load_documents() -> list:
    docs = []
    for file_path in glob.glob(f"{DATA_DIR}/*"):
        if file_path.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file_path.endswith(".json"):
            loader = JSONLoader(file_path, jq_schema=".")
        else:
            continue

        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"[ERROR] Failed to load {file_path}: {e}")
    return docs

def setup_vector_store(docs):
    if not docs:
        raise ValueError("No documents loaded.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("No chunks were created from documents.")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)
    vectordb.save_local(DB_FAISS_PATH)
    return vectordb

def get_vector_store():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    index_path = os.path.join(DB_FAISS_PATH, "index.faiss")
    if os.path.exists(index_path):
        return FAISS.load_local(DB_FAISS_PATH, embeddings=embeddings)
    else:
        docs = load_documents()
        return setup_vector_store(docs)

def preview_files() -> List[str]:
    previews = []
    for file_path in glob.glob(f"{DATA_DIR}/*"):
        previews.append(os.path.basename(file_path))
    return previews
