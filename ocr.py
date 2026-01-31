# OCR module for receipt text extraction and structuring
# Supports PaddleOCR (hybrid) and Gemini-only modes

import os
import json
import pandas as pd
import google.generativeai as genai
import PIL.Image
from datetime import datetime
from paddleocr import PaddleOCR

# Initialize PaddleOCR once (lazy loading)
_paddle_ocr = None


def _get_paddle_ocr():
    """Lazy initialization of PaddleOCR to avoid slow startup."""
    global _paddle_ocr
    if _paddle_ocr is None:
        _paddle_ocr = PaddleOCR(lang='th', use_angle_cls=True)
    return _paddle_ocr


def extract_text_paddle(image_path: str) -> str:
    """Extract raw text from image using PaddleOCR."""
    ocr = _get_paddle_ocr()
    result = ocr.ocr(image_path)

    if not result or not result[0]:
        return ""

    lines = []
    for line in result[0]:
        text = line[1][0]
        confidence = line[1][1]
        if confidence > 0.5:
            lines.append(text)

    return '\n'.join(lines)


def structure_with_gemini(text: str) -> pd.DataFrame:
    """Use Gemini to structure extracted text into a DataFrame."""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    prompt = f"""Analyze this 7-Eleven receipt text and return a JSON LIST of objects with these keys:
"Timestamp", "Item", "Category", "Price", "Size".

Format Timestamp as YYYY-MM-DD HH:MM.
If you can't find a date, use '{datetime.now().strftime("%Y-%m-%d")} 12:00'.
Price should be a number (no currency symbols).
Category should be one of: Food, Beverage, Snack, Household, Other.

Receipt text:
{text}

Return ONLY valid JSON, no markdown formatting."""

    response = model.generate_content(prompt)
    clean_json = response.text.replace('```json', '').replace('```', '').strip()
    data = json.loads(clean_json)

    return pd.DataFrame(data)


def run_gemini_only_ocr(image_path: str) -> pd.DataFrame:
    """Original approach: send image directly to Gemini."""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    img = PIL.Image.open(image_path)
    prompt = f"""Analyze this 7-Eleven receipt. Return a JSON LIST of objects with these keys:
"Timestamp", "Item", "Category", "Price", "Size".

Format Timestamp as YYYY-MM-DD HH:MM.
If you can't find a date, use '{datetime.now().strftime("%Y-%m-%d")} 12:00'.
Price should be a number (no currency symbols).
Category should be one of: Food, Beverage, Snack, Household, Other.

Return ONLY valid JSON, no markdown formatting."""

    response = model.generate_content([prompt, img])
    clean_json = response.text.replace('```json', '').replace('```', '').strip()
    data = json.loads(clean_json)

    return pd.DataFrame(data)


def run_hybrid_ocr(image_path: str) -> tuple[pd.DataFrame, str]:
    """
    Hybrid approach: PaddleOCR extracts text, Gemini structures it.
    Returns (DataFrame, extracted_text) for debugging.
    """
    raw_text = extract_text_paddle(image_path)
    if not raw_text:
        raise ValueError("PaddleOCR could not extract any text from the image")

    df = structure_with_gemini(raw_text)
    return df, raw_text


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has required columns and proper types."""
    if 'Timestamp' not in df.columns:
        df['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M")

    df['Timestamp'] = df['Timestamp'].fillna(datetime.now().strftime("%Y-%m-%d %H:%M"))
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    for col in ['Item', 'Category', 'Size']:
        if col not in df.columns:
            df[col] = ''

    if 'Price' not in df.columns:
        df['Price'] = 0.0

    return df
