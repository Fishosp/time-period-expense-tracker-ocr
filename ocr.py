# OCR module for receipt text extraction and structuring
# Supports EasyOCR + Gemini/Ollama for text structuring

import os
import json
import pandas as pd
import google.generativeai as genai
import PIL.Image
import requests
from datetime import datetime
import easyocr

# Initialize EasyOCR once (lazy loading)
_easyocr_reader = None


def _get_easyocr_reader():
    """Lazy initialization of EasyOCR to avoid slow startup."""
    global _easyocr_reader
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(['th', 'en'], gpu=False)
    return _easyocr_reader


def extract_text_easyocr(image_path: str) -> str:
    """Extract raw text from image using EasyOCR."""
    reader = _get_easyocr_reader()
    result = reader.readtext(image_path)

    if not result:
        return ""

    lines = []
    for detection in result:
        text = detection[1]
        confidence = detection[2]
        if confidence > 0.3:
            lines.append(text)

    return '\n'.join(lines)


def _get_structuring_prompt(text: str) -> str:
    """Generate the prompt for structuring receipt text."""
    return f"""Analyze this 7-Eleven receipt text and return a JSON LIST of objects with these keys:
"Timestamp", "Item", "Category", "Price", "Size".

Format Timestamp as YYYY-MM-DD HH:MM.
If you can't find a date, use '{datetime.now().strftime("%Y-%m-%d")} 12:00'.
Price should be a number (no currency symbols).
Category should be one of: Food, Beverage, Snack, Household, Other.

Receipt text:
{text}

Return ONLY valid JSON, no markdown formatting."""


def structure_with_gemini(text: str) -> pd.DataFrame:
    """Use Gemini to structure extracted text into a DataFrame."""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    prompt = _get_structuring_prompt(text)
    response = model.generate_content(prompt)
    clean_json = response.text.replace('```json', '').replace('```', '').strip()
    data = json.loads(clean_json)

    return pd.DataFrame(data)


def structure_with_ollama(text: str, model: str = "llama3.2") -> pd.DataFrame:
    """Use Ollama to structure extracted text into a DataFrame."""
    ollama_url = os.environ.get('OLLAMA_URL', 'http://localhost:11434')

    prompt = _get_structuring_prompt(text)

    response = requests.post(
        f'{ollama_url}/api/generate',
        json={
            'model': model,
            'prompt': prompt,
            'stream': False
        },
        timeout=120
    )

    if response.status_code != 200:
        raise ValueError(f"Ollama error: {response.text}")

    result = response.json()
    raw_response = result.get('response', '')
    clean_json = raw_response.replace('```json', '').replace('```', '').strip()
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


def run_hybrid_ocr(image_path: str, llm_backend: str = "gemini", ollama_model: str = "llama3.2") -> tuple[pd.DataFrame, str]:
    """
    Hybrid approach: EasyOCR extracts text, LLM structures it.
    Returns (DataFrame, extracted_text) for debugging.

    Args:
        image_path: Path to the receipt image
        llm_backend: "gemini" or "ollama"
        ollama_model: Model name for Ollama (ignored if using Gemini)
    """
    raw_text = extract_text_easyocr(image_path)
    if not raw_text:
        raise ValueError("EasyOCR could not extract any text from the image")

    if llm_backend == "ollama":
        df = structure_with_ollama(raw_text, ollama_model)
    else:
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
