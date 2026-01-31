# time-period-expense-tracker-ocr

A Streamlit app that uses Google's Gemini AI to extract structured data from 7-Eleven receipts. Upload a receipt image and the AI will parse items, prices, categories, and timestamps - then track your spending over time.

## Features

- OCR-powered receipt scanning using Gemini AI
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

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Get a Gemini API key from https://aistudio.google.com/app/apikey

```bash
cp .env.example .env
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

## Sample Images

The `samples/` folder contains example 7-Eleven receipt images for testing.
