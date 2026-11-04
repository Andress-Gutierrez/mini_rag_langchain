"""Rutas y settings del mini RAG (Ollama local)."""

from pathlib import Path

from dotenv import load_dotenv

# Raíz del proyecto (mini_rag_langchain/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
COLLECTION_NAME = "politicas_demo"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# Retrieval
TOP_K = 3

# Ollama local (sin API key de pago)
# API: http://localhost:11434  — el servicio Ollama debe estar corriendo
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"
