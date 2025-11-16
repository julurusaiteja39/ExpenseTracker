# Personal Finance Advisor with OCR (Agentic RAG)

This project is an end-to-end demo you can put on your resume:

- **Backend**: FastAPI + LangChain + LangGraph + RAG + FAISS vector store
- **Frontend**: Simple React UI (CDN) for uploading receipts and asking questions
- **NLP / AI**:
  - OCR on receipt images (Pytesseract)
  - Lightweight NLP parsing of receipts
  - Vectorization of transactions with embeddings
  - Retrieval-Augmented Generation (RAG) over your past spending
  - Agentic workflow with LangGraph (multi-step graph: retrieve → analyze → answer)

## Folder structure

- `backend/` – FastAPI app and LangGraph workflow
- `frontend/` – Plain React frontend (open `index.html` directly)

---

## 1. Backend setup

From the `backend` folder:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate   # on Windows
# source .venv/bin/activate  # on macOS / Linux

pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file:

```bash
copy .env.example .env   # Windows
# or: cp .env.example .env
```

Edit `.env` and set your OpenAI key (and optionally model names):

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Run the backend:

```bash
uvicorn backend.main:app --reload
```

It should start on `http://127.0.0.1:8000`.

- Test health:
  - Open `http://127.0.0.1:8000/` in the browser
- API docs:
  - `http://127.0.0.1:8000/docs`

---

## 2. Frontend

The frontend is intentionally very simple. No bundler needed.

```bash
cd frontend
```

Just open `index.html` in your browser (double-click or open via VS Code "Open with Live Server").

By default, it talks to the backend at `http://127.0.0.1:8000`.

Features:

- Upload a receipt image → backend runs OCR + NLP → stores a parsed transaction + adds to vector store
- Ask natural language questions like:
  - "How much did I spend on groceries last month?"
  - "What are my biggest spending categories?"
  - "How much did I spend on Uber last week?"
- Backend uses:
  - FAISS vector store + embeddings (vectorization)
  - LangGraph workflow:
    - `retrieve` node: retrieve relevant transactions with RAG
    - `analyze` node: summarize spending pattern
    - `answer` node: generate final answer + tips

---

## 3. Good phrases for your resume

- Built an **agentic AI Personal Finance Advisor** using **LangGraph** and **LangChain**.
- Implemented **RAG** over a **FAISS vector store** of OCR'd receipts (transaction embeddings with OpenAI).
- Designed a multi-step **LangGraph workflow** for question understanding, retrieval, analysis, and answer generation.
- Integrated **OCR (Pytesseract)** and lightweight **NLP parsing** to extract structured transactions from raw receipt images.
- Exposed the system via a **FastAPI backend** and a minimal **React frontend** for interactive analysis.