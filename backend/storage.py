import json
import uuid
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from .config import DATA_DIR, OPENAI_EMBEDDING_MODEL
from .models import Transaction

TRANSACTIONS_PATH = DATA_DIR / "transactions.jsonl"
VECTORSTORE_PATH = DATA_DIR / "vectorstore"
VECTOR_INDEX_FILE = VECTORSTORE_PATH / "index.faiss"
VECTOR_META_FILE = VECTORSTORE_PATH / "index.pkl"

_vectorstore: Optional[FAISS] = None


def _ensure_files():
    DATA_DIR.mkdir(exist_ok=True)
    if not TRANSACTIONS_PATH.exists():
        TRANSACTIONS_PATH.write_text("")
    VECTORSTORE_PATH.mkdir(exist_ok=True)
_ensure_files()


def load_transactions() -> List[Transaction]:
    txs: List[Transaction] = []
    if not TRANSACTIONS_PATH.exists():
        return txs
    with TRANSACTIONS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                txs.append(Transaction(**obj))
            except Exception:
                continue
    return txs


def append_transaction(tx: Transaction) -> None:
    with TRANSACTIONS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(tx.model_dump()) + "\n")


def _transaction_to_doc(tx: Transaction) -> Document:
    text_parts = [
        f"Transaction ID: {tx.id}",
        f"Date: {tx.date or 'unknown'}",
        f"Merchant: {tx.merchant or 'unknown'}",
        f"Category: {tx.category or 'uncategorized'}",
        f"Amount: {tx.amount or 'unknown'} {tx.currency or ''}".strip(),
        "",
        f"Raw Text: {tx.raw_text}",
    ]
    content = "\n".join(text_parts)
    metadata = {
        "id": tx.id,
        "date": tx.date,
        "merchant": tx.merchant,
        "category": tx.category,
        "amount": tx.amount,
        "currency": tx.currency,
    }
    return Document(page_content=content, metadata=metadata)


def get_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

    def _rebuild_empty():
        # start with placeholder so retrieval works
        vs = FAISS.from_texts(
            texts=["This is a placeholder transaction. Add receipts to build your personal finance memory."],
            embedding=embeddings,
            metadatas=[{"id": "placeholder"}],
        )
        VECTORSTORE_PATH.mkdir(exist_ok=True)
        vs.save_local(str(VECTORSTORE_PATH))
        return vs

    # Attempt to load existing if both index files are present; otherwise rebuild.
    if VECTOR_INDEX_FILE.exists() and VECTOR_META_FILE.exists():
        try:
            _vectorstore = FAISS.load_local(
                str(VECTORSTORE_PATH),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            return _vectorstore
        except Exception as exc:
            print(f"[WARN] Failed to load existing vectorstore, rebuilding. Error: {exc}")

    # Build from existing transactions if any
    txs = load_transactions()
    docs = [_transaction_to_doc(tx) for tx in txs] if txs else []
    if docs:
        _vectorstore = FAISS.from_documents(docs, embeddings)
    else:
        _vectorstore = _rebuild_empty()
        return _vectorstore

    VECTORSTORE_PATH.mkdir(exist_ok=True)
    _vectorstore.save_local(str(VECTORSTORE_PATH))
    return _vectorstore


def add_transaction_to_vectorstore(tx: Transaction) -> None:
    vs = get_vectorstore()
    doc = _transaction_to_doc(tx)
    vs.add_documents([doc])
    VECTORSTORE_PATH.mkdir(exist_ok=True)
    vs.save_local(str(VECTORSTORE_PATH))


def reset_data() -> None:
    """Delete all stored transactions and vectorstore artifacts."""
    global _vectorstore
    _vectorstore = None
    if TRANSACTIONS_PATH.exists():
        TRANSACTIONS_PATH.unlink(missing_ok=True)  # empty log
    if VECTORSTORE_PATH.exists():
        shutil.rmtree(VECTORSTORE_PATH, ignore_errors=True)
    _ensure_files()  # recreate empty structures


def create_transaction_from_parsed(parsed: Dict[str, Any], raw_text: str) -> Transaction:
    return Transaction(
        id=str(parsed.get("id") or uuid.uuid4()),
        date=parsed.get("date"),
        merchant=parsed.get("merchant"),
        category=parsed.get("category"),
        amount=parsed.get("amount"),
        currency=parsed.get("currency") or "USD",
        raw_text=raw_text,
    )
