# OCR module for receipt text extraction and structuring
# Supports EasyOCR + Gemini/Ollama for text structuring

import os
import json
import logging
import pandas as pd
import google.generativeai as genai
import PIL.Image
import requests
from datetime import datetime
import easyocr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize EasyOCR once (lazy loading)
_easyocr_reader = None


def _get_easyocr_reader():
    """Lazy initialization of EasyOCR to avoid slow startup."""
    global _easyocr_reader
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(['th', 'en'], gpu=False)
    return _easyocr_reader


def extract_text_easyocr(image_path: str) -> str:
    """Extract raw text from image using EasyOCR, preserving line structure."""
    logger.info(f"Extracting text from: {image_path}")
    reader = _get_easyocr_reader()
    result = reader.readtext(image_path)

    if not result:
        logger.warning("EasyOCR returned no results")
        return ""

    # Group text by vertical position to preserve table structure
    # Each detection: (bbox, text, confidence)
    # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    items = []
    for detection in result:
        bbox = detection[0]
        text = detection[1]
        confidence = detection[2]
        if confidence > 0.3:
            # Use average y position of top edge as line indicator
            y_pos = (bbox[0][1] + bbox[1][1]) / 2
            x_pos = bbox[0][0]  # Left edge x position
            items.append((y_pos, x_pos, text))

    if not items:
        return ""

    # Sort by y position, then x position
    items.sort(key=lambda x: (x[0], x[1]))

    # Group items that are on the same line (within threshold)
    line_threshold = 20  # pixels
    lines = []
    current_line = []
    current_y = items[0][0]

    for y_pos, x_pos, text in items:
        if abs(y_pos - current_y) < line_threshold:
            current_line.append((x_pos, text))
        else:
            # Sort current line by x position and add to lines
            current_line.sort(key=lambda x: x[0])
            lines.append(' '.join([t[1] for t in current_line]))
            current_line = [(x_pos, text)]
            current_y = y_pos

    # Don't forget the last line
    if current_line:
        current_line.sort(key=lambda x: x[0])
        lines.append(' '.join([t[1] for t in current_line]))

    extracted = '\n'.join(lines)
    logger.info(f"Extracted text:\n{extracted}")
    return extracted


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
    logger.info(f"Calling Gemini")
    response = model.generate_content(prompt)
    raw_response = response.text
    logger.info(f"Gemini raw response:\n{raw_response}")

    clean_json = _extract_json_from_response(raw_response)

    try:
        data = json.loads(clean_json)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        raise ValueError(f"Could not parse Gemini response as JSON. Raw response:\n{raw_response[:500]}")

    return pd.DataFrame(data)


def _extract_json_from_response(text: str) -> str:
    """Extract JSON array from LLM response, handling various formats."""
    import re

    # Remove markdown code blocks
    text = text.replace('```json', '').replace('```', '').strip()

    # Try to find JSON array in the response
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        return match.group(0)

    # Try to find JSON object and wrap in array
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        return f'[{match.group(0)}]'

    return text


def structure_with_ollama(text: str, model: str = "llama3.2") -> pd.DataFrame:
    """Use Ollama to structure extracted text into a DataFrame."""
    ollama_url = os.environ.get('OLLAMA_URL', 'http://localhost:11434')

    prompt = _get_structuring_prompt(text)
    logger.info(f"Calling Ollama model: {model}")
    logger.info(f"Prompt:\n{prompt}")

    response = requests.post(
        f'{ollama_url}/api/generate',
        json={
            'model': model,
            'prompt': prompt,
            'stream': False
        },
        timeout=120
    )

    logger.info(f"Ollama response status: {response.status_code}")

    if response.status_code != 200:
        logger.error(f"Ollama error response: {response.text}")
        raise ValueError(f"Ollama error: {response.text}")

    result = response.json()
    raw_response = result.get('response', '')
    logger.info(f"Ollama raw response:\n{raw_response}")

    if not raw_response.strip():
        raise ValueError("Ollama returned empty response. Try a different model or check if Ollama is running.")

    clean_json = _extract_json_from_response(raw_response)
    logger.info(f"Extracted JSON:\n{clean_json}")

    try:
        data = json.loads(clean_json)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        raise ValueError(f"Could not parse LLM response as JSON. Raw response:\n{raw_response[:500]}")

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
