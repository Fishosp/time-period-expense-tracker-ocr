# Receipt OCR Expense Tracker
# Uses EasyOCR + LLM to extract structured data from receipt images

import streamlit as st
import pandas as pd
import requests
from dotenv import load_dotenv


def get_ollama_models():
    """Fetch available models from Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [m["name"] for m in models]
    except Exception:
        pass
    return ["qwen2.5:7b"]  # fallback

st.set_page_config(layout="wide", page_title="Expense Tracker")

from ocr import (
    run_hybrid_ocr,
    run_gemini_only_ocr,
    normalize_dataframe,
)
from components.review_queue import init_review_state, render_review_queue

load_dotenv()

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("⚙️ Settings")

    ocr_method = st.radio(
        "OCR Method",
        ["Hybrid (EasyOCR + LLM)", "Gemini Only"],
        index=0,
        help="Hybrid uses EasyOCR for text extraction, then an LLM for structuring."
    )

    llm_backend = st.radio(
        "LLM Backend",
        ["Groq (Free API)", "Ollama (Local)", "Gemini (API)"],
        index=0,
        help="Groq: free cloud API. Ollama: local, offline. Gemini: Google API."
    )

    ollama_model = None
    groq_model = None

    if "Ollama" in llm_backend:
        available_models = get_ollama_models()
        default_index = 0
        if "qwen2.5:7b" in available_models:
            default_index = available_models.index("qwen2.5:7b")
        ollama_model = st.selectbox(
            "Ollama Model",
            options=available_models,
            index=default_index,
            help="Select from locally available models"
        )
    elif "Groq" in llm_backend:
        groq_models = [
            "qwen/qwen-3-32b",
            "llama-3.3-70b-versatile",
            "deepseek-r1-distill-llama-70b",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
        ]
        groq_model = st.selectbox(
            "Groq Model",
            options=groq_models,
            index=0,
            help="Qwen 3 32B recommended for multilingual/Thai"
        )

    st.divider()

    st.subheader("Admin Tools")
    st.info("Use this if the app feels 'stuck' or shows old data.")

    if st.button("🗑️ Hard Reset App"):
        st.session_state.clear()
        st.cache_data.clear()
        st.rerun()

# --- INITIALIZATION ---
if 'master_db' not in st.session_state:
    st.session_state.master_db = pd.DataFrame(columns=["Timestamp", "Item", "Category", "Price", "Size"])

# --- OCR FUNCTION ---
def run_ocr(image_path: str, method: str, llm: str, ollama_model: str = None, groq_model: str = None):
    """Run OCR with selected method and return normalized DataFrame."""
    try:
        if "Ollama" in llm:
            backend = "ollama"
        elif "Groq" in llm:
            backend = "groq"
        else:
            backend = "gemini"

        if "Hybrid" in method:
            df, raw_text = run_hybrid_ocr(
                image_path,
                llm_backend=backend,
                ollama_model=ollama_model or "qwen2.5:7b",
                groq_model=groq_model or "qwen/qwen-3-32b"
            )
            st.session_state.last_raw_text = raw_text
        else:
            df = run_gemini_only_ocr(image_path)
            st.session_state.last_raw_text = None

        return normalize_dataframe(df)

    except Exception as e:
        st.error(f"⚠️ Parsing Error: {e}")
        return None

# --- MAIN UI ---
st.title("📅 Time-Period Expense Tracker")

uploaded_file = st.file_uploader("Upload Receipt", type=["jpg", "png"])

if uploaded_file:
    # Image on left (smaller), table on right (larger)
    col1, col2 = st.columns([1, 2])

    with col1:
        header_col, button_col = st.columns([2, 1])
        with header_col:
            st.subheader("🖼️ Receipt")
        with button_col:
            button_placeholder = st.empty()

        st.image(uploaded_file, width="stretch")

        analyze_clicked = button_placeholder.button("🚀 Analyze")
        if analyze_clicked:
            button_placeholder.empty()
            with button_placeholder:
                with st.spinner("Extracting..." if "Hybrid" in ocr_method else "Reading..."):
                    with open("temp.jpg", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    df = run_ocr("temp.jpg", ocr_method, llm_backend, ollama_model, groq_model)
                    if df is not None:
                        init_review_state(df)
            st.rerun()

    with col2:
        if 'review_items' in st.session_state:
            accepted_df = render_review_queue()

            if accepted_df is not None:
                # User clicked "Accept Selected"
                st.session_state.master_db = pd.concat([st.session_state.master_db, accepted_df]).drop_duplicates()
                st.success(f"Saved {len(accepted_df)} items to history!")
                # Clear review state
                if "review_items" in st.session_state:
                    del st.session_state.review_items
                if "editing_item" in st.session_state:
                    del st.session_state.editing_item
                if "last_raw_text" in st.session_state:
                    del st.session_state.last_raw_text
                st.rerun()
        else:
            st.info("👆 Upload a receipt and click Analyze")

# --- TIME-PERIOD ANALYSIS ---
if not st.session_state.master_db.empty:
    st.divider()
    st.subheader("🕒 Spending Over Time")

    time_data = st.session_state.master_db.copy()
    time_data['Date'] = time_data['Timestamp'].dt.date
    daily_spend = time_data.groupby('Date')['Price'].sum()

    st.line_chart(daily_spend)

    with st.expander("📄 View Full Transaction History"):
        st.dataframe(st.session_state.master_db.sort_values("Timestamp", ascending=False))

# --- CATEGORY SUMMARY ---
if not st.session_state.master_db.empty:
    st.divider()
    st.header("📊 Spending Summary by Category")

    df_clean = st.session_state.master_db.copy()
    df_clean["Price"] = pd.to_numeric(df_clean["Price"], errors='coerce').fillna(0)

    # Exclude total/summary rows (Thai and English keywords)
    exclude_keywords = [
        "ยอดสุทธิ", "ยอดรวม", "รวมทั้งสิ้น", "ชำระเงิน", "ทรูวอลเล็ท",
        "Total", "Subtotal", "Tax", "VAT", "Payment", "Change", "Cash",
        "ตราปั๊มจังหวัด", "ภารกิจช้อปครบ", "รับเงินภารกิจช้อปหน้าร้าน", "สิทธิ์แลกซื้อสุดคุ้มAMB", "บริการ"
    ]

    mask = df_clean["Item"].str.contains('|'.join(exclude_keywords), na=False)
    df_items_only = df_clean[~mask]

    category_summary = df_items_only.groupby("Category")["Price"].sum().reset_index()

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("Summary Table")
        st.dataframe(category_summary, width="stretch", hide_index=True)

    with col_b:
        st.subheader("Spending Chart")
        st.bar_chart(data=category_summary, x="Category", y="Price")

    total_spent = float(category_summary["Price"].sum())
    st.metric("Total Overall Spending", f"{total_spent:,.2f} THB")
