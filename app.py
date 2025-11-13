# app.py
# Al-Jazeera Premium UI + Fade/Slide animations + Gold Ring Gauge + Fact-check accuracy
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import time
import random
import logging
from typing import Optional, Tuple, List
from ftfy import fix_text
from urllib.parse import urljoin, quote_plus

# ML similarity utilities (already listed in requirements)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Preserve API key logic (use st.secrets if available)
API_KEY = st.secrets["FACTCHECK_API_KEY"] if hasattr(st, "secrets") and "FACTCHECK_API_KEY" in st.secrets else None

# ---------------------------
# Keep model / scraping helpers intact (no logic change)
# ---------------------------
def clean(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    try:
        s = fix_text(s)
    except Exception:
        pass
    return " ".join(s.split()).strip()

# Google Fact Check helper (kept behavior)
def get_fact_check_results(query: str, max_results: int = 10):
    if not API_KEY:
        return []
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "key": API_KEY, "pageSize": max_results}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        claims = data.get("claims", [])
        formatted = []
        for claim in claims:
            text = claim.get("text", "") or ""
            for r in claim.get("claimReview", []):
                formatted.append({
                    "claim_text": text,
                    "publisher": r.get("publisher", {}).get("name", "Unknown"),
                    "title": r.get("title") or r.get("url") or "",
                    "rating": r.get("textualRating") or "",
                    "url": r.get("url") or ""
                })
        return formatted
    except Exception as e:
        return [{"claim_text": "", "publisher": "Error", "title": str(e), "rating": "", "url": ""}]

# ---------------------------
# Accuracy computation
# ---------------------------
def rating_to_numeric(rating: str) -> Optional[float]:
    """
    Map textualRating to numeric in [0,1]:
    - True-like -> 1.0
    - Mostly True / Half True -> 0.8 / 0.6 approximations
    - No Rating or ambiguous -> None
    - False-like -> 0.0
    """
    if not rating:
        return None
    r = rating.lower()
    if "pants" in r or "false" in r or "no" in r and "true" not in r:
        return 0.0
    if "true" in r and ("mostly" not in r and "half" not in r):
        return 1.0
    if "mostly true" in r:
        return 0.85
    if "half true" in r or "half flip" in r or "half" in r:
        return 0.65
    if "barely true" in r:
        return 0.35
    if "flip" in r:
        return 0.5
    # fallback None for ambiguous labels
    return None

def compute_fact_accuracy(user_claim: str, results: List[dict]) -> dict:
    """
    Compute:
      - similarity scores (0..1) between user_claim and each returned claim_text/title
      - verdict numeric (0..1) where available
      - combined credibility: weighted average (similarity * 0.7 + verdict * 0.3) when verdict exists,
        else fallback to similarity-only score.
    Returns dict with overall percentage and details per-match.
    """
    out = {"overall_pct": 0.0, "details": []}
    if not results:
        return out

    corpus = [user_claim.strip()] + [ (r.get("claim_text","") + " " + (r.get("title") or "")) for r in results ]
    # TF-IDF + cosine similarity
    try:
        tf = TfidfVectorizer(stop_words='english', lowercase=True)
        tfidf = tf.fit_transform(corpus)
        cosines = linear_kernel(tfidf[0:1], tfidf[1:]).flatten()  # similarities with user_claim
    except Exception:
        # fallback to simple ratio using difflib if TF-IDF fails
        from difflib import SequenceMatcher
        cosines = []
        for r in results:
            combined = (r.get("claim_text","") + " " + (r.get("title") or ""))
            if not combined.strip():
                cosines.append(0.0)
            else:
                s = SequenceMatcher(None, user_claim.lower(), combined.lower()).ratio()
                cosines.append(s)

    detail_rows = []
    combined_scores = []
    for i, r in enumerate(results):
        sim = float(cosines[i]) if i < len(cosines) else 0.0
        verdict_num = rating_to_numeric(r.get("rating","") or "")
        if verdict_num is None:
            combined = sim  # rely solely on similarity
        else:
            combined = 0.7 * sim + 0.3 * verdict_num
        detail_rows.append({
            "index": i,
            "publisher": r.get("publisher",""),
            "title": r.get("title",""),
            "rating": r.get("rating",""),
            "url": r.get("url",""),
            "similarity": sim,
            "verdict_num": verdict_num,
            "combined": combined
        })
        combined_scores.append(combined)

    # overall score: max combined score (best matching fact-check) scaled to percentage
    if combined_scores:
        best_score = float(np.max(combined_scores))
        overall_pct = round(best_score * 100, 2)
    else:
        overall_pct = 0.0

    out["overall_pct"] = overall_pct
    out["details"] = sorted(detail_rows, key=lambda x: x["combined"], reverse=True)
    return out

# ---------------------------
# Styling + animations + gauge creation (UI only)
# ---------------------------
GAUGE_SVG_TEMPLATE = """
<div class="gauge-wrap">
  <svg viewBox="0 0 120 120" class="gauge">
    <defs>
      <linearGradient id="grad" x1="0" x2="1">
        <stop offset="0%" stop-color="#f6e0b5"/>
        <stop offset="100%" stop-color="#d6b672"/>
      </linearGradient>
      <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
        <feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#b88a3a" flood-opacity="0.15"/>
      </filter>
    </defs>

    <circle class="g-bg" cx="60" cy="60" r="48" stroke-width="10" fill="none" />
    <circle class="g-bar" cx="60" cy="60" r="48" stroke-width="10" stroke="url(#grad)" stroke-linecap="round" fill="none"
            style="stroke-dasharray: {circ}; stroke-dashoffset: {offset}; transform-origin:60px 60px;" />
    <text x="60" y="62" class="g-text">{percent}%</text>
  </svg>
</div>
"""

CSS_AND_ANIM = """
<style>
/* Page-level */
:root{
  --aj-gold: #c99b46;
  --aj-dark: #222;
  --aj-muted: #6b6b6b;
  --panel-bg: #fbfaf8;
  --accent-ink: #5a4a2f;
}
body { -webkit-font-smoothing:antialiased; }

/* topbar */
.topbar {
  background: linear-gradient(90deg, rgba(201,155,70,0.98), rgba(196,156,78,0.94));
  padding:18px;border-radius:10px;margin-bottom:14px;box-shadow:0 8px 28px rgba(34,34,34,0.08);
  color: #fff;
  transform: translateY(-12px);
  animation: slideDown 550ms ease-out forwards;
}
@keyframes slideDown {
  to { transform: translateY(0); }
}

/* page fade */
.page-fade { opacity: 0; transform: translateY(6px); animation: pageFadeIn 700ms ease-out forwards; }
@keyframes pageFadeIn {
  to { opacity:1; transform: translateY(0); }
}

/* panels/cards */
.panel {
  background: #fff;
  padding:14px;
  border-radius:10px;
  border:1px solid rgba(34,34,34,0.04);
  box-shadow: 0 6px 20px rgba(34,34,34,0.03);
  transition: transform 220ms ease, box-shadow 220ms ease;
}
.card {
  background: linear-gradient(180deg,#fff,#fbfbfb);
  padding:12px;border-radius:8px;margin-bottom:12px;border:1px solid rgba(34,34,34,0.04);
  transition: transform 220ms ease, box-shadow 220ms ease;
}
.card:hover { transform: translateY(-8px); box-shadow: 0 18px 40px rgba(34,34,34,0.08); }

/* buttons */
.btn-gold {
  background: linear-gradient(90deg,#d6b672,#b5842b);
  color: #fff; padding:10px 14px; border-radius:8px; border:none; font-weight:600;
  box-shadow: 0 8px 22px rgba(181,131,54,0.12);
  transition: transform 130ms ease, box-shadow 130ms ease;
}
.btn-gold:active { transform: translateY(2px); }

/* subtle ripple on click via focus */
.btn-gold:focus { outline: none; box-shadow: 0 0 0 6px rgba(201,155,70,0.12); }

/* gauge styling */
.gauge-wrap { display:flex; align-items:center; justify-content:center; width:160px; height:160px; margin:auto; }
.gauge { width:160px; height:160px; }
.g-bg { stroke: rgba(34,34,34,0.06); }
.g-bar {
  stroke-dasharray: 301.44;
  stroke-dashoffset: 301.44;
  transition: stroke-dashoffset 1200ms cubic-bezier(.22,.9,.2,1);
  filter: url(#shadow);
}
.g-text { font-size:20px; font-weight:700; fill: #5a4627; text-anchor:middle; font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial; }

/* result list reveal */
.result-item { opacity: 0; transform: translateY(8px); animation: itemIn 520ms ease-out forwards; }
@keyframes itemIn { to { opacity:1; transform: translateY(0); } }

/* small helper */
.meta { color: var(--aj-muted); font-size:0.92rem; }
.headline { font-weight:700; color: var(--aj-dark); margin-bottom:6px; }
.small { font-size:0.9rem; color: #777; }

/* responsive tweaks */
@media (max-width: 760px){
  .gauge-wrap { width:130px; height:130px; }
  .gauge { width:130px; height:130px; }
}
</style>
"""

# ---------------------------
# Streamlit App UI
# ---------------------------
def app():
    st.set_page_config(page_title='AI vs. Fact — Premium', layout='wide')
    st.markdown(CSS_AND_ANIM, unsafe_allow_html=True)

    st.markdown('<div class="topbar"><h1 style="margin:0;font-size:1.9rem">Global FactCheck Network</h1><div style="opacity:.95;font-weight:500">Al-Jazeera inspired • Premium editorial UI</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="page-fade">', unsafe_allow_html=True)
    tabs = st.tabs(["Home", "Scraper", "Model Bench", "Fact Check"])

    # HOME
    with tabs[0]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Welcome — Premium Animated FactCheck Portal")
        st.write("This portal scrapes Politifact, trains NLP comparators and cross-checks claims using verified fact-check archives.\n\nNow with animated UI and a gold circular credibility gauge.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ===============================
# SCRAPER TAB
# ===============================
with tabs[1]:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.header("Politifact Scraper")

    st.write("Scrape Politifact fact-checks by selecting a date range.")

    min_date = pd.to_datetime('2007-01-01')
    max_date = pd.to_datetime('today').normalize()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=pd.to_datetime('2023-01-01'))
    with col2:
        end_date = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Scrape Politifact Data"):
        if start_date > end_date:
            st.error("Start Date must be before or equal to End Date.")
        else:
            with st.spinner("Scraping Politifact…"):
                scraped_df = scrape_data_by_date_range(
                    pd.to_datetime(start_date),
                    pd.to_datetime(end_date)
                )

            if scraped_df.empty:
                st.warning("No data scraped — try a different date range.")
            else:
                st.success(f"Scraping complete! {len(scraped_df)} records extracted.")
                st.session_state['scraped_df'] = scraped_df

                st.download_button(
                    "Download CSV",
                    scraped_df.to_csv(index=False).encode('utf-8'),
                    file_name="politifact_scraped.csv",
                    mime="text/csv"
                )

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# MODEL BENCH TAB
# ===============================
with tabs[2]:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.header("Model Bench")

    if 'scraped_df' not in st.session_state:
        st.info("No scraped data found. Please scrape data first from the Scraper tab.")
    else:
        df = st.session_state['scraped_df']
        st.write(f"Loaded dataset with **{len(df)}** items.")

        phases = ["Lexical & Morphological", "Syntactic", "Semantic", "Discourse", "Pragmatic"]
        selected_phase = st.selectbox("Choose Feature Set:", phases)

        if st.button("Run Benchmark"):
            with st.spinner("Training and evaluating models…"):
                df_results = evaluate_models(df, selected_phase, NLP_MODEL)

            if df_results.empty:
                st.warning("Model training failed or returned no results.")
            else:
                st.success("Benchmark Complete!")
                st.dataframe(df_results, use_container_width=True)

                st.session_state['df_results'] = df_results
                st.session_state['selected_phase_run'] = selected_phase

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# FACT CHECK TAB (gauge fixed)
# ===============================
with tabs[3]:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.header("Cross-Platform Fact Check — Animated Gauge")
    st.write("Enter a claim to verify it against global fact-check archives.")

    col1, col2 = st.columns([3,1])
    with col1:
        user_claim = st.text_area("Enter claim / headline to verify", height=120)
        max_results = st.slider("Max results to retrieve", min_value=3, max_value=15, value=8)

    with col2:
        st.markdown("<div style='text-align:center;margin-top:6px'><strong>Credibility Meter</strong></div>",
                    unsafe_allow_html=True)

        r = 48.0
        circ = 2 * np.pi * r
        offset = circ

        init_svg = GAUGE_SVG_TEMPLATE.format(
            circ=f"{circ:.2f}",
            offset=f"{offset:.2f}",
            percent="--"
        )
        st.markdown(init_svg, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Run Fact Check"):
        if not user_claim.strip():
            st.warning("Please enter a claim first.")
        else:
            with st.spinner("Checking claim…"):
                results = get_fact_check_results(user_claim, max_results)
                computed = compute_fact_accuracy(user_claim, results)

            overall_pct = computed.get("overall_pct", 0.0)
            details = computed.get("details", [])

            offset = circ * (1 - overall_pct / 100.0)
            offset = max(0.0, min(offset, circ))

            svg = GAUGE_SVG_TEMPLATE.format(
                circ=f"{circ:.2f}",
                offset=f"{offset:.2f}",
                percent=f"{overall_pct:.1f}"
            )
            st.markdown(svg, unsafe_allow_html=True)

            st.markdown(f"### Overall Credibility Accuracy: **{overall_pct:.1f}%**")
            st.markdown("<hr>", unsafe_allow_html=True)

            if not details:
                st.info("No fact-check results found.")
            else:
                best = details[0]

                st.subheader("Best Match")
                st.write(f"**{best['title']}**")
                st.write(f"Publisher: {best['publisher']}")
                st.write(f"Verdict: {best['rating']}")
                st.write(f"Similarity: {best['similarity']:.3f}")
                st.write(f"Combined Score: {best['combined']:.3f}")

                st.markdown("---")
                st.subheader("All Matches")

                for d in details:
                    st.write(f"### {d['title']}")
                    st.write(f"Publisher: {d['publisher']}")
                    st.write(f"Verdict: {d['rating']}")
                    st.write(f"Similarity: {d['similarity']:.3f}")
                    st.write(f"Combined Score: {d['combined']:.3f}")
                    if d["url"]:
                        st.write(f"[Open Article]({d['url']})")
                    st.markdown("---")

    st.markdown("</div>", unsafe_allow_html=True)



if __name__ == "__main__":
    app()
