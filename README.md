# time-period-expense-tracker-ocr

A Streamlit app that uses PaddleOCR and Google's Gemini AI to extract structured data from 7-Eleven receipts. Upload a receipt image and the AI will parse items, prices, categories, and timestamps - then track your spending over time.

## Features

- **Hybrid OCR** - PaddleOCR for Thai text extraction + Gemini for structuring
- **Fallback mode** - Gemini-only OCR available via sidebar toggle
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

> **Note:** PaddleOCR will download ~150MB of models on first run. This is normal and only happens once.

### 4. Configure your API key

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

The app supports two OCR methods (selectable in the sidebar):

| Mode | How it works | Best for |
|------|--------------|----------|
| **Hybrid** (default) | PaddleOCR extracts text → Gemini structures it | Thai receipts, better accuracy |
| **Gemini Only** | Image sent directly to Gemini | Quick testing, fallback option |

Hybrid mode also shows the raw extracted text in an expandable section for debugging.

## Sample Images

The `samples/` folder contains example 7-Eleven receipt images for testing.

## Troubleshooting

**PaddleOCR installation issues:**
- On some systems, you may need to install PaddlePaddle separately first:
  ```bash
  pip install paddlepaddle
  pip install paddleocr
  ```
- For GPU support, see [PaddlePaddle installation guide](https://www.paddlepaddle.org.cn/install/quick)

**Slow first run:**
- PaddleOCR downloads models (~150MB) on first use. Subsequent runs are faster.
