import os
import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import OPENAI_API_KEY
from .models import AskQuestionRequest, AskQuestionResponse, UploadReceiptResponse, Transaction
from .graph import build_workflow
from .ocr import extract_text, simple_parse_receipt
from .storage import (
    load_transactions,
    append_transaction,
    add_transaction_to_vectorstore,
    create_transaction_from_parsed,
    reset_data,
)

if not OPENAI_API_KEY:
    # This will fail at runtime if you actually call OpenAI, but it's a nice explicit message.
    print("[WARN] OPENAI_API_KEY is not set. LLM calls will fail until you configure it.")

app = FastAPI(
    title="Personal Finance Advisor with OCR (Agentic RAG)",
    version="0.1.0",
    description="Agentic AI powered personal finance assistant built with FastAPI, LangGraph, LangChain, RAG, and OCR.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

workflow = build_workflow()
graph_app = workflow.compile()


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Personal Finance Advisor backend is running"}


@app.get("/transactions", response_model=list[Transaction])
def list_transactions():
    return load_transactions()


@app.post("/ask", response_model=AskQuestionResponse)
def ask_question(payload: AskQuestionRequest):
    state = {"question": payload.question}
    final_state = graph_app.invoke(state)

    return AskQuestionResponse(
        answer=final_state.get("answer", "Sorry, I could not generate an answer."),
        retrieved_context=final_state.get("retrieved_context", ""),
        analysis_points=final_state.get("analysis_points", []),
        tips=final_state.get("tips", []),
    )


@app.post("/upload_receipt", response_model=UploadReceiptResponse)
async def upload_receipt(file: UploadFile = File(...)):
    if not file:
        return JSONResponse(status_code=400, content={"detail": "No file uploaded."})

    contents = await file.read()
    try:
        ocr_text = extract_text(contents, content_type=file.content_type, filename=file.filename)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    except Exception as exc:  # pragma: no cover
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to process the uploaded file: {exc}"},
        )
    parsed = simple_parse_receipt(ocr_text)
    if not parsed.get("date"):
        # Fallback to upload date when invoice date is missing
        parsed["date"] = datetime.date.today().isoformat()

    tx = create_transaction_from_parsed(parsed, raw_text=ocr_text)
    append_transaction(tx)
    add_transaction_to_vectorstore(tx)

    return UploadReceiptResponse(
        ocr_text=ocr_text,
        parsed_transaction=tx.model_dump(),
    )


@app.post("/reset_data")
def reset_all_data():
    reset_data()
    return {"status": "ok", "message": "All transactions and vector store cleared."}
