# mini_rag_langchain

End-to-end **RAG (Retrieval-Augmented Generation)** pipeline built with **LangChain + Ollama** — fully local, zero cost, no OpenAI key required.

```
document → chunks → embeddings (nomic-embed-text) → Chroma
      → question → retrieve → LLM (llama3.2 via local Ollama) → answer
```

**Stack:** Python 3.10 · LangChain · Chroma (local vector store) · Ollama (`llama3.2` + `nomic-embed-text`)

**What it does:** indexes a document, retrieves the most relevant chunks for a question, and answers using a local LLM. The retrieved chunks are printed on every query, so the retrieval step can be verified instead of trusted blindly.

**Why it matters:** it demonstrates the core pieces of a RAG system — chunking, embeddings, vector search and grounded generation — without paying for API calls.

> 🇪🇸 La documentación en español está más abajo, a partir de "## Requisitos".

---

## Requisitos

- **Python 3.10** (versión del proyecto; verificada en máquina del autor: **3.10.5**)
  - Archivo `.python-version` → `3.10`
  - Creación de venv con el launcher: `py -3.10 -m venv .venv`
- [Ollama](https://ollama.com) instalado y **corriendo**
- Modelos:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
```

Comprobar API local:

```powershell
Invoke-RestMethod -Uri "http://localhost:11434/api/tags"
```

## Arranque rápido

En PowerShell, desde esta carpeta:

```powershell
# Usar exactamente Python 3.10 (no 3.11/3.12 si también los tienes)
py -3.10 --version
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
# debe salir Python 3.10.x

python -m pip install --upgrade pip
pip install -r requirements.txt
```

`pip install` con Chroma/LangChain puede tardar varios minutos en la fase _Installing collected packages_ (normal en Windows).

No hace falta `OPENAI_API_KEY`. Ollama debe estar abierto/servicio activo.

### 1) Indexar (alimentar la base vectorial)

```powershell
python -m src.ingest
```

La primera vez de embeddings puede tardar un poco (carga del modelo en RAM).

### 2) Preguntar (chat en terminal)

```powershell
python -m src.chat
```

Cada pregunta muestra:

1. Los **chunks** recuperados (retrieval)
2. La **respuesta** de `llama3.2` local

Escribe `salir` para terminar.

## Preguntas de prueba

| Pregunta                                | Resultado esperado              |
| --------------------------------------- | ------------------------------- |
| ¿Cuántos días tengo para un reembolso?  | 30 días desde la entrega        |
| ¿Qué pasa si cuesta más de 200 dólares? | Escala a humano                 |
| ¿Se reembolsa un curso digital?         | No (productos digitales)        |
| ¿De qué color es el logo?               | “No lo sé…” (no está en el doc) |

## Cómo se conecta a Ollama

```
Python (LangChain)
  ChatOllama / OllamaEmbeddings
        │
        ▼  HTTP local
  http://localhost:11434
        │
        ▼
  Ollama  →  llama3.2 + nomic-embed-text
```

- El modelo queda **instalado en disco** (`ollama list`).
- Mientras el **servicio Ollama** esté activo, el código puede usarlo.
- No hace falta dejar `ollama run` abierto; se carga al pedir.

Config en `src/config.py`: `OLLAMA_BASE_URL`, `CHAT_MODEL`, `EMBEDDING_MODEL`.

## Estructura

```
mini_rag_langchain/
├── data/politicas_reembolso.md
├── src/config.py
├── src/ingest.py
├── src/chat.py
├── vectorstore/          # generado (gitignored)
├── .python-version       # 3.10
├── .env.example          # opcional (no hace falta key)
├── requirements.txt
└── README.md
```

## Qué es / qué no es

| Sí                    | No (aún)                      |
| --------------------- | ----------------------------- |
| LangChain             | LangGraph                     |
| Chroma local          | Cloud vector DB               |
| Ollama local (gratis) | OpenAI de pago (en este mini) |
| Chat CLI              | Streamlit / web               |

## Si reindexas

Si algún día cambias de proveedor de embeddings, **borra y vuelve a correr** `python -m src.ingest` (el ingest ya limpia `vectorstore/`).

## Costos

**$0** en uso normal: todo corre en tu PC vía Ollama.
