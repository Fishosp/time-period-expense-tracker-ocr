# 7-Eleven Receipt OCR Expense Tracker
# Uses EasyOCR + Gemini AI to extract structured data from receipt images

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from ocr import (
    run_hybrid_ocr,
    run_gemini_only_ocr,
    normalize_dataframe,
)

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
        ["Ollama (Local)", "Gemini (API)"],
        index=0,
        help="Ollama runs locally (free, offline). Gemini requires API key."
    )

    if "Ollama" in llm_backend:
        ollama_model = st.text_input(
            "Ollama Model",
            value="llama3.2",
            help="Model name (e.g., llama3.2, mistral, gemma2)"
        )
    else:
        ollama_model = None

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
def run_ocr(image_path: str, method: str, llm: str, model: str = None):
    """Run OCR with selected method and return normalized DataFrame."""
    try:
        backend = "ollama" if "Ollama" in llm else "gemini"

        if "Hybrid" in method:
            df, raw_text = run_hybrid_ocr(image_path, llm_backend=backend, ollama_model=model or "llama3.2")
            st.session_state.last_raw_text = raw_text
        else:
            df = run_gemini_only_ocr(image_path)
            st.session_state.last_raw_text = None

        return normalize_dataframe(df)

    except Exception as e:
        st.error(f"⚠️ OCR Error: {e}")
        return None

# --- MAIN UI ---
st.title("📅 Time-Period Expense Tracker")

uploaded_file = st.file_uploader("Upload Receipt", type=["jpg", "png"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🖼️ Uploaded Receipt")
        st.image(uploaded_file, use_container_width=True)

    with col2:
        st.subheader("🔍 OCR Verification")
        if st.button("🚀 Analyze Receipt"):
            with st.spinner("Extracting text..." if "Hybrid" in ocr_method else "AI is reading..."):
                with open("temp.jpg", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.session_state.current_scan = run_ocr("temp.jpg", ocr_method, llm_backend, ollama_model)

        if 'current_scan' in st.session_state and st.session_state.current_scan is not None:
            st.info("💡 Edit the table below to correct any AI errors.")

            # Show raw extracted text for hybrid mode (debugging)
            if st.session_state.get('last_raw_text'):
                with st.expander("📝 Raw OCR Text (EasyOCR)"):
                    st.text(st.session_state.last_raw_text)

            edited_results = st.data_editor(
                st.session_state.current_scan,
                use_container_width=True,
                key="ocr_editor"
            )

            if st.button("💾 Save to History"):
                st.session_state.master_db = pd.concat([st.session_state.master_db, edited_results]).drop_duplicates()
                st.success("Data saved to Master File!")
                del st.session_state.current_scan
                if 'last_raw_text' in st.session_state:
                    del st.session_state.last_raw_text
                st.rerun()

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

    # Exclude total/summary rows from Thai receipts
    exclude_keywords = [
        "ยอดสุทธิ", "ยอดรวม", "Total", "ชำระเงิน", "ทรูวอลเล็ท", "รวมทั้งสิ้น",
        "ตราปั๊มจังหวัด", "ภารกิจช้อปครบ", "รับเงินภารกิจช้อปหน้าร้าน", "สิทธิ์แลกซื้อสุดคุ้มAMB", "บริการ"
    ]

    mask = df_clean["Item"].str.contains('|'.join(exclude_keywords), na=False)
    df_items_only = df_clean[~mask]

    category_summary = df_items_only.groupby("Category")["Price"].sum().reset_index()

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("Summary Table")
        st.dataframe(category_summary, use_container_width=True, hide_index=True)

    with col_b:
        st.subheader("Spending Chart")
        st.bar_chart(data=category_summary, x="Category", y="Price")

    total_spent = float(category_summary["Price"].sum())
    st.metric("Total Overall Spending", f"{total_spent:,.2f} THB")
