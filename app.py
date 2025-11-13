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

# Preserve API key logic
API_KEY = st.secrets["FACTCHECK_API_KEY"] if hasattr(st, "secrets") and "FACTCHECK_API_KEY" in st.secrets else None


# ---------------------------
# CLEAN
# ---------------------------
def clean(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    try:
        s = fix_text(s)
    except Exception:
        pass
    return " ".join(s.split()).strip()


# ---------------------------
# GOOGLE FACT CHECK API
# ---------------------------
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
# RATING MAP
# ---------------------------
def rating_to_numeric(rating: str) -> Optional[float]:
    if not rating:
        return None
    r = rating.lower()
    if "pants" in r or "false" in r:
        return 0.0
    if "mostly true" in r:
        return 0.85
    if "half" in r:
        return 0.65
    if "barely true" in r:
        return 0.35
    if "true" in r:
        return 1.0
    return None


# ---------------------------
# ACCURACY COMPUTATION
# ---------------------------
def compute_fact_accuracy(user_claim: str, results: List[dict]) -> dict:
    out = {"overall_pct": 0.0, "details": []}
    if not results:
        return out

    corpus = [user_claim] + [(r["claim_text"] + " " + r["title"]) for r in results]

    try:
        tf = TfidfVectorizer(stop_words='english')
        tfidf = tf.fit_transform(corpus)
        cosines = linear_kernel(tfidf[0:1], tfidf[1:]).flatten()
    except:
        from difflib import SequenceMatcher
        cosines = [SequenceMatcher(None, user_claim.lower(), (r["claim_text"] + r["title"]).lower()).ratio()
                   for r in results]

    details = []
    combined_scores = []

    for i, r in enumerate(results):
        sim = cosines[i]
        verdict = rating_to_numeric(r["rating"])
        if verdict is None:
            combined = sim
        else:
            combined = 0.7 * sim + 0.3 * verdict

        details.append({
            "publisher": r["publisher"],
            "title": r["title"],
            "rating": r["rating"],
            "similarity": sim,
            "combined": combined,
            "url": r["url"]
        })
        combined_scores.append(combined)

    best = max(combined_scores)
    out["overall_pct"] = round(best * 100, 2)
    out["details"] = sorted(details, key=lambda x: x["combined"], reverse=True)
    return out


# ---------------------------
# POLITIFACT SCRAPER
# ---------------------------
def scrape_data_by_date_range(start_date, end_date):
    base = "https://www.politifact.com/factchecks/"
    all_rows = []
    cur = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    while cur <= end_date:
        url = f"{base}{cur.year}/{cur.month}/{cur.day}/"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                cur += pd.Timedelta(days=1)
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            items = soup.select("li.m-statement")

            for item in items:
                quote = item.select_one(".m-statement__quote")
                speaker = item.select_one(".m-statement__name")
                meter = item.select_one(".m-statement__meter img")
                link = item.select_one("a")

                all_rows.append({
                    "date": str(cur.date()),
                    "statement": clean(quote.text if quote else ""),
                    "speaker": clean(speaker.text if speaker else ""),
                    "rating": meter["alt"] if meter else "",
                    "url": "https://www.politifact.com" + link["href"] if link else ""
                })
        except:
            pass

        cur += pd.Timedelta(days=1)

    return pd.DataFrame(all_rows)


# ---------------------------
# GAUGE SVG
# ---------------------------
GAUGE_SVG_TEMPLATE = """
<div class="gauge-wrap">
  <svg viewBox="0 0 120 120" class="gauge">
    <circle class="g-bg" cx="60" cy="60" r="48" stroke-width="10" fill="none" />
    <circle class="g-bar" cx="60" cy="60" r="48" stroke-width="10" stroke="url(#grad)" stroke-linecap="round"
      fill="none" style="stroke-dasharray:{circ}; stroke-dashoffset:{offset};" />
    <defs>
      <linearGradient id="grad">
        <stop offset="0%" stop-color="#f6e0b5"/>
        <stop offset="100%" stop-color="#d6b672"/>
      </linearGradient>
    </defs>
    <text x="60" y="64" font-size="22" text-anchor="middle" fill="#6c5528">{percent}%</text>
  </svg>
</div>
"""


# ---------------------------
# MAIN APP
# ---------------------------
def app():

    st.set_page_config(page_title="AI vs Fact Premium", layout="wide")

    # STYLES
    st.markdown("""
    <style>
    .topbar {background:#c9a24a;padding:18px;border-radius:12px;color:white;margin-bottom:20px;}
    .panel {background:white;padding:18px;border-radius:10px;border:1px solid #eee;
            box-shadow:0 6px 18px rgba(0,0,0,0.04);}
    .gauge-wrap{display:flex;justify-content:center;margin-top:10px;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="topbar">
        <h1 style='margin:0;'>Global FactCheck Network</h1>
        <p style='margin:0;'>Premium UI • Al-Jazeera Inspired</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------
    # FIXED TABS (inside app)
    # ---------------------
    tabs = st.tabs(["Home", "Scraper", "Model Bench", "Fact Check"])

    # HOME TAB
    with tabs[0]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Welcome")
        st.write("Premium fact-checking portal.")
        st.markdown('</div>', unsafe_allow_html=True)

    # SCRAPER TAB
    with tabs[1]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.header("Politifact Scraper")

        min_date = pd.to_datetime("2007-01-01")
        max_date = pd.to_datetime("today")

        c1, c2 = st.columns(2)
        with c1:
            s = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=pd.to_datetime("2023-01-01"))
        with c2:
            e = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

        if st.button("Scrape Politifact Data"):
            if s > e:
                st.error("Start date must be before end date.")
            else:
                with st.spinner("Scraping..."):
                    df = scrape_data_by_date_range(s, e)
                if df.empty:
                    st.warning("No data found.")
                else:
                    st.success(f"Scraped {len(df)} items.")
                    st.dataframe(df)
                    st.download_button("Download CSV", df.to_csv(index=False).encode(),
                                       "politifact.csv", "text/csv")

        st.markdown('</div>', unsafe_allow_html=True)

    # MODEL BENCH TAB
    with tabs[2]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.header("Model Bench")
        st.info("Model training disabled in this UI version.")
        st.markdown('</div>', unsafe_allow_html=True)

    # FACT CHECK TAB
    with tabs[3]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.header("Cross-Platform Fact Check — Animated Gauge")

        c1, c2 = st.columns([3, 1])
        with c1:
            claim = st.text_area("Enter claim", height=120)
            maxr = st.slider("Max results", 3, 15, 8)

        with c2:
            r = 48
            circ = round(2 * np.pi * r, 2)
            offset = circ
            init_svg = GAUGE_SVG_TEMPLATE.format(circ=circ, offset=offset, percent="--")
            st.markdown(init_svg, unsafe_allow_html=True)

        if st.button("Run Fact Check"):
            if not claim.strip():
                st.warning("Enter a claim.")
            else:
                with st.spinner("Checking..."):
                    results = get_fact_check_results(claim, maxr)
                    comp = compute_fact_accuracy(claim, results)

                pct = comp["overall_pct"]
                details = comp["details"]

                offset = circ * (1 - pct / 100)

                svg = GAUGE_SVG_TEMPLATE.format(circ=circ, offset=offset, percent=pct)
                st.markdown(svg, unsafe_allow_html=True)

                st.subheader(f"Overall Accuracy: {pct}%")
                st.markdown("---")

                for d in details:
                    st.write(f"### {d['title']}")
                    st.write(f"Publisher: {d['publisher']}")
                    st.write(f"Rating: {d['rating']}")
                    st.write(f"Similarity: {d['similarity']:.3f}")
                    st.write(f"Combined Score: {d['combined']:.3f}")
                    if d["url"]:
                        st.write(f"[Open Article]({d['url']})")
                    st.markdown("---")

        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    app()
