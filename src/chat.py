"""
Consulta (puntos del pipeline al preguntar):

  pregunta → búsqueda en Chroma → prompt + contexto → ChatOllama (llama3.2) → respuesta

Ejecutar desde la raíz del proyecto:

  python -m src.chat

Requisitos: Ollama corriendo + modelos:
  ollama pull nomic-embed-text
  ollama pull llama3.2
  python -m src.ingest   # al menos una vez
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from src.config import (
    CHAT_MODEL,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    TOP_K,
    VECTORSTORE_DIR,
)

SYSTEM_PROMPT = """Eres un asistente de soporte de Tienda Demo.

Responde SOLO con hechos del contexto de abajo. Lee TODOS los chunks.

DISTINCIONES IMPORTANTES (no las mezcles):
- "Plazo" / "cuántos días para un reembolso" / "días para solicitar" / "pedir devolución"
  → es el tiempo para PRESENTAR la solicitud (en la política: 30 días desde la ENTREGA).
- "Tiempo de procesamiento" / "cuándo me devuelven el dinero" / "días hábiles del reembolso"
  → es el tiempo DESPUÉS de aprobado para devolver el dinero (en la política: 3 a 5 días hábiles).
- Si el usuario pregunta por días del reembolso sin aclarar, asume PLAZO DE SOLICITUD (no procesamiento),
  salvo que pregunte explícitamente por la devolución del dinero o el procesamiento.

Reglas:
1) Usa el apartado correcto del contexto. No digas "3 a 5 días" si preguntan cuánto tienen para solicitar.
2) Si CUALQUIER chunk tiene el dato correcto para esa pregunta, úsalo. No digas "no lo sé" en ese caso.
3) Solo si NINGÚN chunk tiene la información pedida (ej. color del logo), responde exactamente:
   "No lo sé con base en las políticas indexadas."
4) No inventes números ni reglas. Español, breve (1-3 oraciones). Cita qué significa el plazo si hay duda.

Contexto recuperado de las políticas:
{context}
"""


def require_ollama() -> None:
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
            f"Detalle: {exc}"
        )
        sys.exit(1)


def load_vectorstore() -> Chroma:
    if not VECTORSTORE_DIR.exists():
        print(
            "No existe la base vectorial.\n"
            "Primero indexa los documentos:\n"
            "  python -m src.ingest"
        )
        sys.exit(1)

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    return Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def format_context(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(f"[Chunk {i}]\n{doc.page_content}")
    return "\n\n".join(parts)


def answer_question(store: Chroma, llm: ChatOllama, question: str) -> None:
    # 1) Retrieval: la BD vectorial busca chunks parecidos a la pregunta
    docs = store.similarity_search(question, k=TOP_K)

    print("\n--- Chunks recuperados (para que veas el retrieval) ---")
    if not docs:
        print("(ninguno)")
    for i, doc in enumerate(docs, start=1):
        preview = doc.page_content.replace("\n", " ")[:200]
        print(f"[{i}] {preview}...")
    print("-------------------------------------------------------\n")

    # 2) Augment + Generate: contexto al prompt, LLM local redacta
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    messages = prompt.format_messages(
        context=format_context(docs),
        question=question,
    )
    print("(Generando con Ollama... puede tardar en CPU)")
    response = llm.invoke(messages)

    print("Respuesta del LLM:")
    print(response.content)
    print()


def main() -> None:
    require_ollama()
    store = load_vectorstore()
    llm = ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    print("== CHAT RAG (LangChain + Chroma + Ollama) ==")
    print(f"Ollama: {OLLAMA_BASE_URL} | chat: {CHAT_MODEL} | emb: {EMBEDDING_MODEL}")
    print("Escribe una pregunta. 'salir' o 'exit' para terminar.\n")
    print("Ejemplos:")
    print("  - ¿Cuántos días tengo para un reembolso?")
    print("  - ¿Qué pasa si el pedido cuesta más de 200 dólares?")
    print("  - ¿Se reembolsa un curso digital?")
    print("  - ¿Cuál es el color del logo de la tienda?  (debería no saberlo)\n")

    while True:
        try:
            question = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChao.")
            break

        if not question:
            continue
        if question.lower() in {"salir", "exit", "quit", "q"}:
            print("Chao.")
            break

        answer_question(store, llm, question)


if __name__ == "__main__":
    main()
