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

The `samples/` folder contains example receipt images for testing.

## How It Works

### Architecture

The tool uses a **two-stage pipeline** to convert receipt images into structured data:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Receipt   │ ──▶ │   EasyOCR   │ ──▶ │     LLM     │ ──▶ Structured JSON
│    Image    │     │  (Stage 1)  │     │  (Stage 2)  │
└─────────────┘     └─────────────┘     └─────────────┘
                     Text extraction     Data structuring
```

**Stage 1: Text Extraction (EasyOCR)**
- Detects text regions in the image using deep learning
- Recognizes characters using trained language models (Thai + English)
- Preserves spatial layout by grouping text on the same line using bounding box coordinates
- Outputs raw text with line structure preserved

**Stage 2: Data Structuring (LLM)**
- Takes the raw OCR text as input
- Understands context to identify items, prices, dates
- Categorizes products (Food, Beverage, Snack, etc.)
- Fixes common OCR errors (e.g., 'o' misread as '0')
- Outputs clean JSON array of structured items

### Why Two Stages?

**Why not just use an LLM with vision?**
- Vision LLMs (like Gemini) can read images directly, but they're expensive (API costs) and require internet
- Local vision models need powerful GPUs and are slower

**Why not just use OCR?**
- OCR only extracts text - it doesn't understand what's an item, price, or total
- Receipt formats vary wildly between stores and countries
- LLMs excel at understanding context and structure

The hybrid approach gives us the best of both worlds: fast local text extraction + intelligent structuring.

## Asian Language OCR Challenges

### Why Thai (and other Asian languages) are harder to OCR

**1. Character Complexity**
- Thai has 44 consonants, 15 vowels, and 4 tone marks that combine in complex ways
- Characters can stack vertically (vowels above/below consonants)
- No spaces between words - the reader must understand context to segment

```
English:  "Hello World"  (clear word boundaries)
Thai:     "สวัสดีโลก"      (no spaces, vowels wrap around consonants)
```

**2. Font Variations**
- Receipt printers use various fonts, often low-quality thermal printing
- Thai characters are more sensitive to degradation than Latin letters
- Small differences in strokes change meaning entirely

**3. Script Density**
- More visual information packed into the same space
- Similar-looking characters (ก ถ ภ look similar when blurry)
- Numbers can look like letters (0/o, 1/l issues exist in Thai too)

### Why Multilingual Models Matter

Both the OCR stage and the LLM stage need proper multilingual support:

**OCR Stage (EasyOCR)**

EasyOCR uses separate neural network models for each language. When you initialize with `['th', 'en']`:
- Downloads Thai character recognition model (~70MB)
- Downloads English character recognition model (~30MB)
- Runs both models and combines results

Without the Thai model, EasyOCR would:
- Not recognize Thai characters at all
- Output garbage or empty strings
- Miss most of the receipt content

EasyOCR supports 80+ languages. For Asian scripts:
| Language | Script Type | EasyOCR Code |
|----------|-------------|--------------|
| Thai | Abugida | `th` |
| Chinese (Simplified) | Logographic | `ch_sim` |
| Chinese (Traditional) | Logographic | `ch_tra` |
| Japanese | Mixed | `ja` |
| Korean | Alphabetic | `ko` |

**LLM Stage (Ollama/Gemini)**

Even with perfect OCR, the LLM needs multilingual training to:
- Understand that prices are numbers even when OCR misreads "0" as "o"
- Infer meaning from context when characters are ambiguous
- Handle mixed-language text (Thai product names + English brands)
- Know Thai keywords for "total", "tax", "change" to filter them out

**Recommended LLM models for Asian languages:**
| Model | Asian Language Support | Notes |
|-------|------------------------|-------|
| **qwen2.5:7b** | Excellent | Alibaba model, strong CJK + Thai |
| mistral | Good | Decent multilingual, good at JSON |
| llama3.2 | Moderate | English-focused, struggles with Thai |
| gemma2 | Good | Google model, decent Asian support |
