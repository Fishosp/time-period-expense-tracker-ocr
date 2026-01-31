# OCR module for receipt text extraction and structuring
# Supports EasyOCR + Gemini/Ollama/Groq for text structuring

import os
import json
import logging
import pandas as pd
from google import genai
import PIL.Image
import requests
from datetime import datetime
import easyocr
from groq import Groq

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
    return f"""Extract ALL purchased items from this receipt. Return a JSON array.

IMPORTANT: Include EVERY item with a price. Do NOT skip any products.

RULES:
1. Fix OCR errors in prices: 'o'/'O' → '0', 'l'/'I' → '1', 'S' → '5', 'B' → '8'
2. Prices are numbers like 25.00, 119.00
3. ONLY skip: totals, subtotals, tax lines, payment methods, change, promotional text
4. Include ALL product lines - even if names look strange due to OCR errors

JSON format per item:
{{"Timestamp": "YYYY-MM-DD HH:MM", "Item": "name", "Category": "Food|Beverage|Snack|Household|Other", "Price": 0.00, "Size": ""}}

Default date: {datetime.now().strftime("%Y-%m-%d")} 12:00

Receipt:
{text}

Return ONLY valid JSON array with ALL items."""


def structure_with_gemini(text: str) -> pd.DataFrame:
    """Use Gemini to structure extracted text into a DataFrame."""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")

    client = genai.Client(api_key=api_key)

    prompt = _get_structuring_prompt(text)
    logger.info(f"Calling Gemini")
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt
    )
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


def structure_with_ollama(text: str, model: str = "qwen2.5:7b") -> pd.DataFrame:
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


def structure_with_groq(text: str, model: str = "qwen/qwen3-32b") -> pd.DataFrame:
    """Use Groq to structure extracted text into a DataFrame."""
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment")

    client = Groq(api_key=api_key)

    prompt = _get_structuring_prompt(text)
    logger.info(f"Calling Groq model: {model}")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=4096,
    )

    raw_response = response.choices[0].message.content
    logger.info(f"Groq raw response:\n{raw_response}")

    if not raw_response.strip():
        raise ValueError("Groq returned empty response.")

    clean_json = _extract_json_from_response(raw_response)
    logger.info(f"Extracted JSON:\n{clean_json}")

    try:
        data = json.loads(clean_json)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        raise ValueError(f"Could not parse Groq response as JSON. Raw response:\n{raw_response[:500]}")

    return pd.DataFrame(data)


def run_gemini_only_ocr(image_path: str) -> pd.DataFrame:
    """Original approach: send image directly to Gemini."""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")

    client = genai.Client(api_key=api_key)

    # Read image and encode as base64
    with open(image_path, 'rb') as f:
        image_data = f.read()

    import base64
    image_base64 = base64.b64encode(image_data).decode('utf-8')

    prompt = f"""Extract items from this receipt image. Return a JSON array.

Skip totals, subtotals, tax, payment methods, promotional text.
Each product line = one item.

JSON format:
{{"Timestamp": "YYYY-MM-DD HH:MM", "Item": "name", "Category": "Food|Beverage|Snack|Household|Other", "Price": 0.00, "Size": ""}}

Date if not found: {datetime.now().strftime("%Y-%m-%d")} 12:00

Return ONLY valid JSON array."""

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
            prompt,
            genai.types.Part.from_bytes(data=image_data, mime_type='image/jpeg')
        ]
    )
    clean_json = _extract_json_from_response(response.text)
    data = json.loads(clean_json)

    return pd.DataFrame(data)


def run_hybrid_ocr(image_path: str, llm_backend: str = "gemini", ollama_model: str = "qwen2.5:7b", groq_model: str = "qwen/qwen3-32b") -> tuple[pd.DataFrame, str]:
    """
    Hybrid approach: EasyOCR extracts text, LLM structures it.
    Returns (DataFrame, extracted_text) for debugging.

    Args:
        image_path: Path to the receipt image
        llm_backend: "gemini", "ollama", or "groq"
        ollama_model: Model name for Ollama
        groq_model: Model name for Groq
    """
    raw_text = extract_text_easyocr(image_path)
    if not raw_text:
        raise ValueError("EasyOCR could not extract any text from the image")

    if llm_backend == "ollama":
        df = structure_with_ollama(raw_text, ollama_model)
    elif llm_backend == "groq":
        df = structure_with_groq(raw_text, groq_model)
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
