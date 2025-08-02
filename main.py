# main.py

import os
import streamlit as st
from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from utils import save_uploaded_files, load_documents, setup_vector_store, get_vector_store, preview_files
from streamlit_chat import message

# LangGraph Nodes
def retrieve_context(state):
    query = state["query"]
    retriever = state["retriever"]
    docs = retriever.get_relevant_documents(query)
    return {"docs": docs, "query": query, "retriever": retriever}

def generate_answer(state):
    docs = state["docs"]
    query = state["query"]
    context = "\n".join([doc.page_content for doc in docs])
    prompt_template = PromptTemplate.from_template("""
    तुम एक सहायक एआई हो। नीचे दिए गए संदर्भ के आधार पर उपयोगकर्ता के प्रश्न का उत्तर उसी भाषा में दोहराओ, जिस भाषा में प्रश्न पूछा गया हो।

    संदर्भ:
    {context}

    प्रश्न:
    {question}

    उत्तर:
    """)
    llm = Ollama(model="llama3:8b")
    chain = prompt_template | llm
    response = chain.invoke({"context": context, "question": query})
    return {"response": response, "query": query, "retriever": state["retriever"], "docs": docs}

class RAGState(TypedDict):
    query: str
    retriever: Any
    docs: List[Any]
    response: str

def initialize_graph(retriever):
    builder = StateGraph(RAGState)
    builder.add_node("Retriever", retrieve_context)
    builder.add_node("LLM", generate_answer)
    builder.set_entry_point("Retriever")
    builder.add_edge("Retriever", "LLM")
    builder.add_edge("LLM", END)
    return builder.compile()

# Streamlit UI
st.set_page_config(page_title=" Assistant", layout="wide")
st.title("💬 Anshuman here - Ask me anything related to uploaded")

if "graph" not in st.session_state:
    st.session_state.graph = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("🧠 Knowledge Base")
    uploaded_files = st.file_uploader("📤 Upload PDF, TXT, JSON", type=["pdf", "txt", "json"], accept_multiple_files=True)
    if uploaded_files:
        saved = save_uploaded_files(uploaded_files)
        st.success(f"Uploaded {len(saved)} file(s)")
        try:
            docs = load_documents()
            vectordb = setup_vector_store(docs)
            retriever = vectordb.as_retriever()
            st.session_state.graph = initialize_graph(retriever)
            st.session_state.retriever = retriever
            st.success("✅ Vector DB initialized. You can now ask your question.")
        except Exception as e:
            st.error(f"Vector DB build failed: {e}")

    st.markdown("---")
    st.subheader("📄 Uploaded Files")
    for file in preview_files():
        st.write(f"- {file}")

    st.markdown("---")
    st.subheader("🧾 Chat History")
    if st.session_state.history:
        for i, (q, a) in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"Q{i}: {q}"):
                st.markdown(a)

    if st.button("🧹 Clear History"):
        st.session_state.history = []
        st.success("Chat history cleared.")

query = st.text_input("Ask your question")

if query:
    if st.session_state.graph is None or st.session_state.retriever is None:
        st.error("Please upload files to initialize the knowledge base before asking a question.")
    else:
        try:
            output = st.session_state.graph.invoke({"query": query, "retriever": st.session_state.retriever})
            response = output["response"]
            st.session_state.history.append((query, response))

            # Render chat messages
            message(query, is_user=True, key=f"user_{len(st.session_state.history)}")
            message(response, key=f"bot_{len(st.session_state.history)}")
        except Exception as e:
            st.error(f"Failed to get response: {e}")
