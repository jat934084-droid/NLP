import streamlit as st
import requests
from urllib.parse import quote_plus

# --- Preserve existing logic: allow hardcoded API key as before but prefer st.secrets if available ---
DEFAULT_API_KEY = "AIzaSyDmFciPOWcIuxDKilN1WO-SmMkwXUxZrUE"
API_KEY = st.secrets.get("FACTCHECK_API_KEY", DEFAULT_API_KEY) if hasattr(st, "secrets") else DEFAULT_API_KEY

st.set_page_config(page_title="Global FactCheck Network", layout="wide", initial_sidebar_state="collapsed")

# --- Al-Jazeera style light/gold CSS & layout ---
st.markdown(
    """
    <style>
    :root{
        --aj-gold: #c99b46;
        --aj-dark: #232323;
        --panel-bg: #fbfaf8;
    }
    .stApp {
        background: linear-gradient(180deg, var(--panel-bg) 0%, #f6f4f2 100%);
        color: var(--aj-dark);
    }
    .header {
        background: linear-gradient(90deg, rgba(201,155,70,0.98), rgba(196,156,78,0.85));
        padding: 22px;
        border-radius: 10px;
        box-shadow: 0 8px 30px rgba(34,34,34,0.06);
        margin-bottom: 18px;
        color: #fff;
    }
    .header h1 { margin: 0; font-size: 2.2rem; }
    .header p { margin: 4px 0 0 0; color: rgba(255,255,255,0.95); }

    .search-container {
        background: #fff;
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 16px;
        border: 1px solid rgba(34,34,34,0.04);
    }
    .search-input > div > div > input {
        background: #fff !important;
        color: var(--aj-dark) !important;
        border: 1px solid rgba(34,34,34,0.06) !important;
    }
    .search-button > button {
        background: linear-gradient(90deg,#d6b672,#b5842b) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 6px 18px rgba(34,34,34,0.06) !important;
    }

    .card {
        background: linear-gradient(180deg, #ffffff, #fbfbfb);
        padding: 14px;
        border-radius: 10px;
        margin-bottom: 12px;
        border: 1px solid rgba(34,34,34,0.04);
    }
    .card .meta { color: #6b6b6b; font-size: 0.95rem; margin-bottom: 6px; }
    .badge {
        display:inline-block;
        padding:6px 10px;
        border-radius:20px;
        font-weight:600;
        font-size:0.9rem;
        margin-left:8px;
        background: linear-gradient(90deg,#f6e0b5,#f0d28a);
        color: #3b2f1b;
        border: 1px solid rgba(34,34,34,0.05);
    }

    .link-button {
        display:inline-block;
        padding:8px 12px;
        border-radius:8px;
        background: rgba(34,34,34,0.02);
        border: 1px solid rgba(34,34,34,0.04);
        color: var(--aj-dark);
        text-decoration: none;
        margin-top:6px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="header">
        <h1>Global FactCheck Network</h1>
        <p>Editorial verification search — Al-Jazeera inspired UI. Enter a claim or headline and get verified fact-checks.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Search container
st.markdown('<div class="search-container">', unsafe_allow_html=True)
query = st.text_input("", placeholder="Type a claim or headline to verify (e.g. “Country X banned Y”)", key="fact_query")
col1, col2, col3 = st.columns([6,1,1])
with col2:
    check_btn = st.button("Check", key="check_btn", help="Query Google Fact Check Tools API")
with col3:
    clear_btn = st.button("Clear", key="clear_btn")
st.markdown("</div>", unsafe_allow_html=True)

if clear_btn:
    st.session_state["fact_query"] = ""

if check_btn:
    if not query or not query.strip():
        st.warning("Please enter a claim or headline to search.")
    else:
        q = query.strip()
        url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={quote_plus(q)}&key={API_KEY}"
        with st.spinner("Searching verified fact-checks..."):
            try:
                resp = requests.get(url, timeout=15)
                data = resp.json()
            except Exception as e:
                st.error(f"Request failed: {e}")
                data = {}

        if "claims" in data and data["claims"]:
            st.success(f"Found {len(data['claims'])} related claim(s).")
            for claim in data["claims"]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                text = claim.get("text", "—")
                st.markdown(f"### {text}")
                for review in claim.get("claimReview", []):
                    pub = review.get("publisher", {}).get("name", "Unknown")
                    rating = review.get("textualRating", None)
                    rtitle = review.get("title", review.get("url", "")) or ""
                    rurl = review.get("url", "#")
                    st.markdown(f'<div class="meta"><strong>{pub}</strong> <span class="badge">{rating or "No Rating"}</span></div>', unsafe_allow_html=True)
                    if rtitle:
                        st.markdown(f"**{rtitle}**")
                    st.markdown(f'<a class="link-button" target="_blank" href="{rurl}">Read full article</a>', unsafe_allow_html=True)
                    st.markdown("---")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No fact-checks found for this query.")
