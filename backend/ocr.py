import os
from io import BytesIO
from typing import Optional, Dict, Any, List
import re

from PIL import Image, UnidentifiedImageError
import pytesseract
from PyPDF2 import PdfReader

try:
    from pdf2image import convert_from_bytes
except ImportError:  # pragma: no cover - optional dependency
    convert_from_bytes = None

POPPLER_PATH = os.getenv("POPPLER_PATH")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def _looks_like_pdf(content_type: Optional[str], filename: Optional[str], file_bytes: bytes) -> bool:
    if content_type and "pdf" in content_type.lower():
        return True
    if filename and filename.lower().endswith(".pdf"):
        return True
    return file_bytes.startswith(b"%PDF")


def _extract_text_from_image_bytes(image_bytes: bytes) -> str:
    image = Image.open(BytesIO(image_bytes))
    image = image.convert("RGB")
    return pytesseract.image_to_string(image)


def _ocr_pdf_pages(pdf_bytes: bytes) -> str:
    """Fallback OCR for PDFs using pdf2image if textual extraction fails."""
    if convert_from_bytes is None:
        raise ValueError(
            "Unable to extract text from this PDF. Install pdf2image + poppler to OCR scanned PDFs."
        )
    images = convert_from_bytes(pdf_bytes, poppler_path=POPPLER_PATH)
    ocr_segments: List[str] = []
    for img in images:
        buf = BytesIO()
        img.save(buf, format="PNG")
        ocr_segments.append(_extract_text_from_image_bytes(buf.getvalue()))
    return "\n".join(seg.strip() for seg in ocr_segments if seg.strip())


def extract_text(file_bytes: bytes, content_type: Optional[str] = None, filename: Optional[str] = None) -> str:
    """Extract raw text from uploaded content (images or PDFs)."""
    if _looks_like_pdf(content_type, filename, file_bytes):
        return extract_text_from_pdf(file_bytes)

    try:
        return _extract_text_from_image_bytes(file_bytes)
    except UnidentifiedImageError as exc:
        raise ValueError(
            "Unsupported file type. Please upload an image (PNG/JPG/HEIC/etc.) or a PDF."
        ) from exc


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception:
        reader = None

    text_chunks: List[str] = []
    if reader is not None:
        if reader.is_encrypted:
            try:
                reader.decrypt("")  # type: ignore[arg-type]
            except Exception:
                pass
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                text_chunks.append(text)

    combined = "\n".join(chunk.strip() for chunk in text_chunks if chunk.strip())
    if combined:
        return combined

    # If we couldn't read textual content, try OCR via images
    return _ocr_pdf_pages(pdf_bytes)

def categorize_transaction(merchant: str, text: str = "") -> str:
    combined = f"{merchant} {text}".lower()

    categories = {
        "groceries": [
            "costco", "walmart", "aldi", "kroger", "safeway",
            "whole foods", "grocery", "supermarket", "market", "foods"
        ],
        "transport": [
            "uber", "lyft", "taxi", "cab", "fuel", "gas station",
            "shell", "bp", "chevron"
        ],
        "shopping": [
            "amazon", "flipkart", "shopping", "mall", "target",
            "best buy", "electronics"
        ],
        "eating_out": [
            "cafe", "restaurant", "pizza", "burger", "grill",
            "bistro", "coffee", "diner", "bar", "brew"
        ],
        "subscription": [
            "netflix", "spotify", "apple", "aws", "gcp",
            "azure", "prime", "online invoicing", "software",
            "saas", "license"
        ],
        "housing": [
            "rent", "apartments", "property", "hotel", "inn", "villa"
        ],
        "utilities": [
            "electric", "water", "internet", "wifi", "broadband",
            "comcast", "verizon", "att"
        ],
    }

    for cat, keywords in categories.items():
        if any(k in combined for k in keywords):
            return cat

    return "other"


def extract_total_amount(ocr_text: str) -> Optional[float]:
    """Try hard to get the real final total from the receipt text."""
    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]

    # Matches 53.23 or 1,234.56 etc.
    amount_pattern = re.compile(
        r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})|\d+\.\d{2})'
    )

    def find_amount_in_lines(filter_fn):
        for line in lines:
            if not filter_fn(line.lower()):
                continue
            m = amount_pattern.search(line.replace("$", ""))
            if m:
                try:
                    return float(m.group(1).replace(",", ""))
                except ValueError:
                    continue
        return None

    # 1) Best case: line with "total"
    amt = find_amount_in_lines(lambda s: "total" in s)
    if amt is not None:
        return amt

    # 2) Fallback: lines mentioning amount/subtotal/balance
    amt = find_amount_in_lines(
        lambda s: any(k in s for k in ["amount due", "amount", "subtotal", "balance"])
    )
    if amt is not None:
        return amt

    # 3) Last resort: pick the largest "reasonable" amount in the whole text
    candidates: list[float] = []
    for line in lines:
        for m in amount_pattern.findall(line.replace("$", "")):
            try:
                v = float(m.replace(",", ""))
            except ValueError:
                continue
            # filter obviously crazy values
            if 0 < v < 2000:
                candidates.append(v)

    if candidates:
        return max(candidates)

    return None
def detect_currency(ocr_text: str) -> str:
    text = ocr_text.lower()

    # Symbol-based detection
    if "$" in ocr_text:
        return "USD"

    if "€" in ocr_text:
        return "EUR"

    if "£" in ocr_text:
        return "GBP"

    if "₹" in ocr_text or "rs." in text or "inr" in text:
        return "INR"

    if "¥" in ocr_text or "cny" in text or "rmb" in text:
        return "CNY"

    if "cad" in text:
        return "CAD"

    if "aud" in text:
        return "AUD"

    # Word-based (fallback)
    if "usd" in text:
        return "USD"
    if "eur" in text:
        return "EUR"
    if "gbp" in text:
        return "GBP"

    # Final fallback
    return "USD"



def simple_parse_receipt(ocr_text: str) -> Dict[str, Any]:
    """Very lightweight NLP-style parsing using regex heuristics.

    We try to guess total amount, date, and merchant.
    This is intentionally simple but shows NLP / text processing.
    """
    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]

    # ----- Amount: use smarter extractor -----
    amount = extract_total_amount(ocr_text)

    # ----- Date -----
    date = None
    # Capture dates like 2025-11-14, 11/14/2025, 11-14-2025, and 11/14/25
    date_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2}|\d{2}[/-]\d{2}[/-]\d{4}|\d{2}[/-]\d{2}[/-]\d{2})"
    )
    for line in lines:
        m = date_pattern.search(line)
        if m:
            date = m.group(1)
            break

    # ----- Merchant -----
    merchant = None
    for line in lines:
        # first non-empty line that is not mostly digits
        if not re.search(r"\d", line) and len(line) > 2:
            merchant = line
            break

    # ----- Category -----
    category = categorize_transaction(merchant or "", ocr_text)
    currency = detect_currency(ocr_text)
    
    return {
        "date": date,
        "merchant": merchant,
        "category": category,
        "amount": amount,
        "currency": currency,
    }
