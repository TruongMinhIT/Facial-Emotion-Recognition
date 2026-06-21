# utils/style_loader.py

import streamlit as st
from pathlib import Path

def load_css(path: str = "assets/style.css") -> None:
    """Inject CSS tùy chỉnh từ file vào Streamlit."""
    try:
        css_path = Path(path)
        if css_path.exists():
            with open(css_path, encoding="utf-8", errors="ignore") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass  # Không có CSS thì bỏ qua, không crash app