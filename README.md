# time-period-expense-tracker-ocr

A Streamlit app that uses EasyOCR and LLMs to extract structured data from receipts. Upload a receipt image and the AI will parse items, prices, categories, and timestamps - then track your spending over time.

## Features

- **Hybrid OCR** - EasyOCR for Thai/English text extraction + LLM for structuring
- **Ollama support** - Run locally with no API keys (free, offline, private)
- **Gemini support** - Use Google's API as an alternative
- Editable data table to correct AI mistakes
- Spending over time visualization
- Category breakdown with charts
- Transaction history

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Fishosp/time-period-expense-tracker-ocr.git
cd time-period-expense-tracker-ocr
```

### 2. Create and activate a virtual environment

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** EasyOCR will download language models (~100MB for Thai + English) on first run.

### 4. Set up LLM backend

#### Option A: Ollama (Recommended - free, local, offline)

1. Install Ollama from https://ollama.com/download

2. Pull a model:
   ```bash
   ollama pull qwen2.5:7b
   ```

   | Model | Size | Speed | Accuracy | Best for |
   |-------|------|-------|----------|----------|
   | **qwen2.5:7b** | 4.7GB | Medium | High | Recommended - best for receipts and multilingual |
   | deepseek-r1:8b | 4.9GB | Medium | High | Strong reasoning and accuracy |
   | mistral | 4.1GB | Medium | Good | Good instruction following |
   | llama3.2 | 2.0GB | Fast | Moderate | Quick testing, lower memory |
   | gemma2:9b | 5.4GB | Slow | High | Alternative high accuracy |

3. Start Ollama (runs on http://localhost:11434):
   ```bash
   ollama serve
   ```

4. Select your model from the dropdown in the app sidebar

#### Option B: Gemini API

Get a Gemini API key from https://aistudio.google.com/app/apikey

**Linux/Mac:**
```bash
cp .env.example .env
```

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

Edit `.env` and add your API key:

```
GEMINI_API_KEY=your_actual_api_key_here
```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. Upload a receipt image (JPG or PNG)
2. Click "Analyze Receipt" to run OCR
3. Review and edit the extracted data if needed
4. Click "Save to History" to track the transaction
5. View spending analytics in the charts below

## OCR Modes

The app supports different OCR and LLM combinations (selectable in the sidebar):

**OCR Method:**
| Mode | How it works |
|------|--------------|
| **Hybrid** (default) | EasyOCR extracts text → LLM structures it |
| **Gemini Only** | Image sent directly to Gemini |

**LLM Backend:**
| Backend | Pros | Cons |
|---------|------|------|
| **Ollama** (default) | Free, offline, private | Requires local setup |
| **Gemini** | No setup, fast | API limits, requires key |

Hybrid mode shows the raw extracted text in an expandable section for debugging.

## Sample Images

The `samples/` folder contains example 7-Eleven receipt images for testing.
