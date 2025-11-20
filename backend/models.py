from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class AskQuestionRequest(BaseModel):
    question: str = Field(..., description="Natural language finance question")

class AskQuestionResponse(BaseModel):
    answer: str
    retrieved_context: str
    analysis_points: List[str] = Field(default_factory=list, description="Structured spending insights returned by LangGraph.")
    tips: List[str] = Field(default_factory=list, description="Actionable tips suggested by the assistant.")

class UploadReceiptResponse(BaseModel):
    ocr_text: str
    parsed_transaction: Dict[str, Any]

class Transaction(BaseModel):
    id: str
    date: Optional[str] = None
    merchant: Optional[str] = None
    category: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = "USD"
    raw_text: str
