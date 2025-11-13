import streamlit as st
import requests
import time
import pandas as pd
from urllib.parse import quote_plus

# -----------------------------
# ✅ Secure API Key Handling
# -----------------------------
DEFAULT_API_KEY = "AIzaSyDmFciPOWcIuxDKilN1WO-SmMkwXUxZrUE"
API_KEY = st.secrets.get("FACTCHECK_API_KEY", DEFAULT_API_KEY)

st.set_page_config(page_title="Global FactCheck Network", layout="wide", initial_sidebar_state="collapsed")

# -----------------------------
# 🎨 Dark Theme + Modern Layout
# -----------------------------
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #0b1014 0%, #0e1419 100%); color: #E6EEF3; }
.header {
    background: linear-gradient(90deg, rgba(7,54,66,0.9), rgba(139,0,92,0.85));
    padding: 28px; border-radius: 10px; box-shadow: 0 8px 30px rgba(0,0,0,0.6);
    margin-bottom: 18px;
}
.header h1 { margin: 0; font-size: 2.4rem; color: #fff; }
.header p { margin: 4px 0 0 0; color: #d8eaf2; opacity: 0.9; }
.search-container {
    background: rgba(255,255,255,0.03); padding: 18px; border-radius: 8px;
    margin-bottom: 16px; border: 1px solid rgba(255,255,255,0.04);
}
.search-input > div > div > input {
    background: rgba(255,255,255,0.02) !important; color: #e6eef3 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}
.search-button > button {
    background: linear-gradient(90deg,#ff416c,#3f51b5) !important;
    color: white !important; border: none !important;
    box-shadow: 0 6px 18px rgba(63,81,181,0.25) !important;
}
.card {
    background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
    padding: 14px; border-radius: 10px; margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.03);
}
.card .meta { color: #9fb1c6; font-size: 0.9rem; margin-bottom: 6px; }
.badge {
    display:inline-block; padding:6px 10px; border-radius:20px; font-weight:600;
    font-size:0.85rem; margin-left:8px;
}
.badge.TRUE { background: rgba(46,204,113,0.15); color: #2ee071; border: 1px solid rgba(46,204,113,0.25); }
.badge.FALSE { background: rgba(231,76,60,0.12); color: #ff6b6b; border: 1px solid rgba(231,76,60,0.18); }
.badge.NA { background: rgba(255,255,255,0.03); color: #d1dbe6; border: 1px solid rgba(255,255,255,0.03); }
a.link-button {
    display:inline-block; padding:8px 12px; border-radius:8px;
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04);
    color: #dceefc; text-decoration: none; margin-top:6px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>🌐 Global FactCheck Network</h1>
    <p>Verify any news headline or claim against trusted fact-checking sources.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# 🧠 Input Area
# -----------------------------
st.markdown('<div class="search-container">', unsafe_allow_html=True)
query = st.text_input("", placeholder="Type a claim or headline (e.g. 'COVID vaccine banned in Europe')", key="fact_query")
col1, col2, col3 = st.columns([6,1,1])
with col2:
    check_btn = st.button("Check", key="check_btn")
with col3:
    clear_btn = st.button("Clear", key="clear_btn")
st.markdown("</div>", unsafe_allow_html=True)

if clear_btn:
    st.session_state["fact_query"] = ""

# -----------------------------
# ⚙️ Cached API Call
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_fact_check(query):
    """Fetch fact-check results with caching and rate-limit protection."""
    if not API_KEY:
        return {"error": "API key missing. Please configure in Streamlit secrets."}

    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={quote_plus(query)}&key={API_KEY}"
    try:
        time.sleep(0.3)  # small delay to respect rate limits
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# 🔍 Perform Search
# -----------------------------
if check_btn:
    if not query.strip():
        st.warning("Please enter a valid claim or headline.")
    else:
        with st.spinner("Searching verified fact-checks..."):
            data = fetch_fact_check(query.strip())

        if "error" in data:
            st.error(f"API Error: {data['error']}")
        elif "claims" in data and data["claims"]:
            claims = data["claims"]
            st.success(f"✅ Found {len(claims)} verified claim(s).")

            # Collect all results in list for optional CSV export
            export_rows = []

            for claim in claims:
                text = claim.get("text", "No claim text found")
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"### {text}")

                for review in claim.get("claimReview", []):
                    pub = review.get("publisher", {}).get("name", "Unknown")
                    rating = review.get("textualRating", "No Rating")
                    title = review.get("title", review.get("url", ""))
                    rurl = review.get("url", "#")

                    badge_class = "NA"
                    if "TRUE" in rating.upper():
                        badge_class = "TRUE"
                    elif any(k in rating.upper() for k in ["FALSE", "PANTS", "FLIP"]):
                        badge_class = "FALSE"

                    st.markdown(
                        f'<div class="meta"><strong>{pub}</strong> '
                        f'<span class="badge {badge_class}">{rating}</span></div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**{title}**")
                    st.markdown(f'<a class="link-button" target="_blank" href="{rurl}">Read full article</a>', unsafe_allow_html=True)
                    st.markdown("---")

                    export_rows.append({
                        "Claim": text,
                        "Publisher": pub,
                        "Rating": rating,
                        "Article Title": title,
                        "URL": rurl
                    })

                st.markdown("</div>", unsafe_allow_html=True)

            # -----------------------------
            # 📤 CSV Export Button
            # -----------------------------
            df_export = pd.DataFrame(export_rows)
            csv_data = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Fact-Check Results as CSV",
                csv_data,
                file_name="factcheck_results.csv",
                mime="text/csv"
            )
        else:
            st.info("No fact-check results found for this query.")
