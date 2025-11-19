from io import BytesIO
from typing import Optional, Dict, Any
import re

from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text(image_bytes: bytes) -> str:
    """Run OCR on an image and return raw text."""
    image = Image.open(BytesIO(image_bytes))
    # Ensure it's in a format pytesseract likes
    image = image.convert("RGB")
    text = pytesseract.image_to_string(image)
    return text

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
