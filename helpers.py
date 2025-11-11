import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from config import OLLAMA_BASE_URL as OLLAMA_URL, PERSIST_DIR


def load_embeddings(model: str):
    """
    Load Ollama embeddings model (e.g., nomic-embed-text:latest)
    """
    print(f"[Embedding] Loading embedding model: {model}")
    return OllamaEmbeddings(model=model, base_url=OLLAMA_URL)

def load_llm(model: str, temperature: float = 0.3):
    """
    Load Ollama chat model (e.g., qwen3:1.7b)
    """
    print(f"[LLM] Loading model: {model}")
    return ChatOllama(model=model, base_url=OLLAMA_URL, temperature=temperature)

def get_vectorstore(collection: str, namespace: str, embedding_model: str):
    """
    Load Chroma vectorstore using Ollama embeddings
    """
    path = os.path.join(PERSIST_DIR, namespace)
    os.makedirs(path, exist_ok=True)
    embeddings = load_embeddings(embedding_model)
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=path
    )
