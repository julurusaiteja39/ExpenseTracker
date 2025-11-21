# Personal Finance Advisor with OCR (Agentic RAG)

FastAPI + LangGraph + React demo that OCRs receipts, parses transactions, stores them in a FAISS vector store, and answers spend questions with RAG.

- **Backend**: FastAPI, LangChain, LangGraph, FAISS, OpenAI embeddings/LLM, Pytesseract OCR
- **Frontend**: Vite + React, responsive UI for upload/ask/inspect/reset
- **NLP / AI**: OCR, regex-based parsing (amount/merchant/date/category), embeddings + retriever + multi-step LangGraph (retrieve → analyze → answer)
- **Chunked RAG context**: OCR'd receipt text is split into configurable chunks before being embedded so long invoices stay searchable.
- **Structured outputs**: LangGraph nodes use typed schemas so the API also returns structured spending insights and tips alongside the final answer.
- **Universal uploads**: Accepts PDFs and virtually any image type; PDFs use text extraction with OCR fallback (requires pdf2image + Poppler for scanned docs).
- **Quality of life**: Date extractor handles `YYYY-MM-DD`, `MM/DD/YYYY`, `MM-DD-YYYY`, `MM/DD/YY`; falls back to today's upload date if none is found. One-click "Delete all data" to reset transactions/vector store.

## Folder structure

- `backend/` – FastAPI app, OCR, LangGraph workflow, storage utilities
- `frontend/` – Vite + React app (development server and production build)

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

Set your keys/models (plus optional chunking overrides):

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
VECTOR_CHUNK_SIZE=500
VECTOR_CHUNK_OVERLAP=50
POPPLER_PATH=C:\path\to\poppler\bin
```

`VECTOR_CHUNK_SIZE` and `VECTOR_CHUNK_OVERLAP` control how OCR text is chunked before it is embedded in FAISS. Smaller chunks improve recall for long receipts, while larger chunks capture more surrounding context.  
`POPPLER_PATH` is optional but recommended on Windows so `pdf2image` can locate Poppler when OCRing scanned PDFs.

For scanned PDFs you’ll also need the Poppler binaries installed locally. Download them from https://github.com/oschwartz10612/poppler-windows/releases (Windows) or via `brew install poppler` (macOS/Linux) and point `POPPLER_PATH` to the `bin` directory if required.

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

- Via frontend: click "Delete all data" in the Stored transactions card (with confirmation).
- Via API: `POST http://127.0.0.1:8000/reset_data`
- Manual: remove `backend/data/transactions.jsonl` and the vector store folder `backend/data/vectorstore/` (will auto-recreate on next upload).

---
