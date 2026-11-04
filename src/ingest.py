"""
Indexación (puntos del pipeline que alimentan la base vectorial):

  documentos → chunks → embeddings (Ollama) → Chroma

Ejecutar desde la raíz del proyecto:

  python -m src.ingest

Requisitos: Ollama corriendo + modelos:
  ollama pull nomic-embed-text
  ollama pull llama3.2
"""

from __future__ import annotations

import shutil
import sys
import urllib.error
import urllib.request

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    VECTORSTORE_DIR,
)


def require_ollama() -> None:
    """Comprueba que la API local de Ollama responde."""
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status != 200:
                raise OSError(f"status {resp.status}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(
            "No se pudo conectar a Ollama.\n"
            f"  URL: {url}\n"
            "1) Abre la app Ollama (servicio en segundo plano)\n"
            "2) Verifica: ollama list\n"
            "3) Modelos mínimos:\n"
            "     ollama pull nomic-embed-text\n"
            "     ollama pull llama3.2\n"
            f"Detalle: {exc}"
        )
        sys.exit(1)


def load_documents() -> list[Document]:
    """Carga .md y .txt desde data/ (pathlib: el glob {md,txt} falla en Windows)."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta de datos: {DATA_DIR}")

    paths = sorted(
        {
            *DATA_DIR.rglob("*.md"),
            *DATA_DIR.rglob("*.txt"),
            *DATA_DIR.rglob("*.MD"),
            *DATA_DIR.rglob("*.TXT"),
        }
    )
    # Solo archivos (no carpetas)
    paths = [p for p in paths if p.is_file()]

    docs: list[Document] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={"source": str(path)},
            )
        )

    if not docs:
        raise FileNotFoundError(
            f"No hay archivos .md o .txt en {DATA_DIR}. "
            "Agrega al menos un documento de conocimiento."
        )
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return chunks


def build_vectorstore(chunks) -> Chroma:
    # Reindex limpio: borra el store anterior (vectores Ollama ≠ OpenAI)
    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
        collection_name=COLLECTION_NAME,
    )
    return store


def main() -> None:
    require_ollama()

    print("== INGEST (doc → chunks → Ollama embeddings → Chroma) ==")
    print(f"Datos: {DATA_DIR}")
    print(f"Vectorstore: {VECTORSTORE_DIR}")
    print(f"Ollama: {OLLAMA_BASE_URL} | embeddings: {EMBEDDING_MODEL}")

    docs = load_documents()
    print(f"Documentos cargados: {len(docs)}")

    chunks = split_documents(docs)
    print(f"Chunks generados: {len(chunks)} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    for i, chunk in enumerate(chunks, start=1):
        preview = chunk.page_content.replace("\n", " ")[:120]
        source = chunk.metadata.get("source", "?")
        print(f"  [{i}] ({source}) {preview}...")

    print("\nGenerando embeddings con Ollama (puede tardar la 1ª vez)...")
    build_vectorstore(chunks)
    print(f"\nOK — base vectorial guardada en: {VECTORSTORE_DIR}")
    print("Siguiente: python -m src.chat")


if __name__ == "__main__":
    main()
