# RAG_App
In this blog, we explore how to build a document-aware Retrieval-Augmented Generation (RAG) applicationhttps://medium.com/p/5d852ef2fc8c


# macOS
brew install ollama
ollama serve
# Linux (Debian/Ubuntu)
wget https://ollama.com/download/Ollama.zip
unzip Ollama.zip
sudo ./ollama install
# Run the Model
ollama pull llama3:8b
ollama run llama3:8b

# Install Python Dependencies
python3 -m venv .venv
source .venv/bin/activate
pip3 install langchain langgraph faiss-cpu streamlit pypdf pymupdf chromadb nomic


<img width="720" height="986" alt="image" src="https://github.com/user-attachments/assets/794598af-97df-44d7-8a25-8ef92538c302" />


