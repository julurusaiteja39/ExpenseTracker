# Personal Finance Advisor with OCR (Agentic RAG)

FastAPI + LangGraph + React demo that OCRs receipts, parses transactions, stores them in a FAISS vector store, and answers spend questions with RAG.

- **Backend**: FastAPI, LangChain, LangGraph, FAISS, OpenAI embeddings/LLM, Pytesseract OCR
- **Frontend**: Vite + React, responsive UI for upload/ask/inspect/reset
- **NLP / AI**: OCR, regex-based parsing (amount/merchant/date/category), embeddings + retriever + multi-step LangGraph (retrieve → analyze → answer)
- **Quality of life**: Date extractor handles `YYYY-MM-DD`, `MM/DD/YYYY`, `MM-DD-YYYY`, `MM/DD/YY`; falls back to today’s upload date if none is found. One-click “Delete all data” to reset transactions/vector store.

## Folder structure

- `backend/` — FastAPI app, OCR, LangGraph workflow, storage utilities
- `frontend/` — Vite + React app (development server and production build)

---

## 1) Backend setup

From the `backend` folder:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS/Linux

pip install --upgrade pip
pip install -r requirements.txt
```

Create and fill `.env`:

```bash
copy .env.example .env      # Windows
# cp .env.example .env      # macOS/Linux
```

Set your keys/models:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Run the backend:

```bash
uvicorn backend.main:app --reload
```

Check:
- Health: http://127.0.0.1:8000/
- Docs: http://127.0.0.1:8000/docs

---

## 2) Frontend (Vite + React)

From `frontend`:

```bash
npm install
npm run dev      # start dev server (default http://127.0.0.1:5173)
npm run build    # production build to dist/
npm run preview  # serve the production build locally
```

Configuration:
- Backend URL: set `VITE_BACKEND_URL` in a `.env` file in `frontend/` (defaults to `http://127.0.0.1:8000`).
- The UI includes:
  - Upload receipt (OCR + parse + store + embed)
  - Ask question (calls LangGraph workflow)
  - View stored transactions
  - Delete all data button (calls `/reset_data`)

---

## 3) Resetting data

- Via frontend: click “Delete all data” in the Stored transactions card (with confirmation).
- Via API: `POST http://127.0.0.1:8000/reset_data`
- Manual: remove `backend/data/transactions.jsonl` and the vector store folder `backend/data/vectorstore/` (will auto-recreate on next upload).

---
