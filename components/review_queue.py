# Review Queue Component
# Allows users to review, edit, accept/reject extracted receipt items

import streamlit as st
import pandas as pd

CATEGORIES = ["Food", "Beverage", "Snack", "Household", "Other"]


def init_review_state(df: pd.DataFrame) -> None:
    """Initialize review items from DataFrame."""
    items = []
    for idx, row in df.iterrows():
        items.append({
            "id": idx,
            "Timestamp": row.get("Timestamp", ""),
            "Item": row.get("Item", ""),
            "Category": row.get("Category", "Other"),
            "Price": row.get("Price", 0.0),
            "Size": row.get("Size", ""),
            "accepted": True,  # Default to accepted
        })
    st.session_state.review_items = items
    st.session_state.editing_item = None


def render_review_queue() -> pd.DataFrame | None:
    """
    Render the review queue UI.
    Returns DataFrame of accepted items when user confirms, None otherwise.
    """
    if "review_items" not in st.session_state:
        return None

    items = st.session_state.review_items

    # Header
    accepted_count = sum(1 for item in items if item["accepted"])
    total_price = sum(item["Price"] for item in items if item["accepted"])

    st.subheader(f"📋 Review Items ({accepted_count}/{len(items)} selected)")

    # Raw OCR text expander
    if st.session_state.get('last_raw_text'):
        with st.expander("📝 Raw OCR Text"):
            st.text(st.session_state.last_raw_text)

    # Render each item
    for idx, item in enumerate(items):
        _render_item_row(idx, item)

    st.divider()

    # Summary
    st.markdown(f"**Selected:** {accepted_count} items • **Total:** {total_price:,.2f} THB")

    # Batch actions
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

    with col1:
        if st.button("✅ Accept Selected", type="primary", disabled=accepted_count == 0):
            return _get_accepted_dataframe()

    with col2:
        if st.button("☑️ Select All"):
            for item in st.session_state.review_items:
                item["accepted"] = True
            st.rerun()

    with col3:
        if st.button("☐ Deselect All"):
            for item in st.session_state.review_items:
                item["accepted"] = False
            st.rerun()

    with col4:
        if st.button("🗑️ Discard Batch"):
            _clear_review_state()
            st.rerun()

    return None


def _render_item_row(idx: int, item: dict) -> None:
    """Render a single item row with actions."""
    is_accepted = item["accepted"]
    is_editing = st.session_state.get("editing_item") == idx

    # Container styling based on status
    if is_editing:
        _render_edit_form(idx, item)
    else:
        _render_display_row(idx, item, is_accepted)


def _render_display_row(idx: int, item: dict, is_accepted: bool) -> None:
    """Render item in display mode."""
    cols = st.columns([0.5, 3, 1.5, 1.5, 1, 1])

    with cols[0]:
        new_accepted = st.checkbox(
            "Select",
            value=is_accepted,
            key=f"check_{idx}",
            label_visibility="collapsed"
        )
        if new_accepted != is_accepted:
            st.session_state.review_items[idx]["accepted"] = new_accepted
            st.rerun()

    # Apply strikethrough style if rejected
    style = "opacity: 0.4; text-decoration: line-through;" if not is_accepted else ""

    with cols[1]:
        st.markdown(f"<span style='{style}'>{item['Item']}</span>", unsafe_allow_html=True)

    with cols[2]:
        st.markdown(f"<span style='{style}'>{item['Price']:,.2f}</span>", unsafe_allow_html=True)

    with cols[3]:
        st.markdown(f"<span style='{style}'>{item['Category']}</span>", unsafe_allow_html=True)

    with cols[4]:
        if st.button("✏️", key=f"edit_{idx}", help="Edit item"):
            st.session_state.editing_item = idx
            st.rerun()

    with cols[5]:
        if is_accepted:
            if st.button("❌", key=f"reject_{idx}", help="Reject item"):
                st.session_state.review_items[idx]["accepted"] = False
                st.rerun()
        else:
            if st.button("↩️", key=f"undo_{idx}", help="Undo rejection"):
                st.session_state.review_items[idx]["accepted"] = True
                st.rerun()


def _render_edit_form(idx: int, item: dict) -> None:
    """Render item in edit mode."""
    with st.container():
        st.markdown("---")
        st.markdown(f"**Editing Item #{idx + 1}**")

        col1, col2 = st.columns([2, 1])

        with col1:
            new_item = st.text_input("Item Name", value=item["Item"], key=f"edit_item_{idx}")

        with col2:
            new_price = st.number_input("Price", value=float(item["Price"]), key=f"edit_price_{idx}", min_value=0.0, step=0.01)

        col3, col4 = st.columns([1, 1])

        with col3:
            category_idx = CATEGORIES.index(item["Category"]) if item["Category"] in CATEGORIES else 4
            new_category = st.selectbox("Category", CATEGORIES, index=category_idx, key=f"edit_cat_{idx}")

        with col4:
            new_size = st.text_input("Size", value=item["Size"] or "", key=f"edit_size_{idx}")

        btn_col1, btn_col2, _ = st.columns([1, 1, 3])

        with btn_col1:
            if st.button("💾 Save", key=f"save_{idx}"):
                st.session_state.review_items[idx].update({
                    "Item": new_item,
                    "Price": new_price,
                    "Category": new_category,
                    "Size": new_size,
                })
                st.session_state.editing_item = None
                st.rerun()

        with btn_col2:
            if st.button("Cancel", key=f"cancel_{idx}"):
                st.session_state.editing_item = None
                st.rerun()

        st.markdown("---")


def _get_accepted_dataframe() -> pd.DataFrame:
    """Convert accepted items back to DataFrame."""
    accepted = [item for item in st.session_state.review_items if item["accepted"]]

    if not accepted:
        return pd.DataFrame(columns=["Timestamp", "Item", "Category", "Price", "Size"])

    df = pd.DataFrame(accepted)
    df = df[["Timestamp", "Item", "Category", "Price", "Size"]]
    return df


def _clear_review_state() -> None:
    """Clear review-related session state."""
    if "review_items" in st.session_state:
        del st.session_state.review_items
    if "editing_item" in st.session_state:
        del st.session_state.editing_item
    if "current_scan" in st.session_state:
        del st.session_state.current_scan
    if "last_raw_text" in st.session_state:
        del st.session_state.last_raw_text
