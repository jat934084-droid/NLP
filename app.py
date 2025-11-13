# app.py — Full integrated app (UI + Scraper + ML accuracy + FactCheck API)
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, quote_plus
import time
import logging
import os
from typing import Optional, List, Tuple
from ftfy import fix_text

# ML & NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from textblob import TextBlob

# imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# spaCy (attempt to load, but app will show error with instructions if not available)
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# ---------------------------
# Basic config & logger
# ---------------------------
st.set_page_config(page_title="AI Fact-Check — Premium", layout="wide")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------
# API Key handling (no external file)
# ---------------------------
# Optional: set FACTCHECK_API_KEY in Streamlit Secrets or as environment variable
DEFAULT_API_KEY = "AIzaSyDmFciPOWcIuxDKilN1WO-SmMkwXUxZrUE"
API_KEY = None
try:
    API_KEY = st.secrets.get("FACTCHECK_API_KEY", DEFAULT_API_KEY) if hasattr(st, "secrets") else os.environ.get("FACTCHECK_API_KEY", DEFAULT_API_KEY)
except Exception:
    API_KEY = os.environ.get("FACTCHECK_API_KEY", DEFAULT_API_KEY)

# ---------------------------
# Styling (premium UI + sidebar + loaders)
# ---------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
    .stApp { font-family: 'Poppins', sans-serif; background: linear-gradient(135deg,#05060a 0%, #0b1220 100%); color: #e6eef3; }
    .app-title { font-size:44px; font-weight:800; text-align:center; margin-bottom:8px;
                 background: linear-gradient(90deg,#ff3c8c,#6c63ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }

    .glass-card { background: rgba(255,255,255,0.03); border-radius: 16px; padding: 18px; border: 1px solid rgba(255,255,255,0.04); box-shadow: 0 8px 30px rgba(0,0,0,0.6); margin-bottom: 12px; }

    .stButton>button { background: linear-gradient(90deg,#ff3c8c,#6c63ff) !important; color: white !important; padding: 10px 18px; border-radius: 12px; font-weight:700; border: none; box-shadow: 0 8px 20px rgba(108,99,255,0.12); }
    .stButton>button:hover { transform: translateY(-3px); box-shadow: 0 16px 30px rgba(108,99,255,0.18); }

    div[data-baseweb="input"] > div { background: rgba(255,255,255,0.02) !important; border-radius:10px; border:1px solid rgba(255,255,255,0.06) !important; color: #e6eef3 !important; }

    .stTabs [data-baseweb="tab"] { padding: 10px 18px; border-radius: 10px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04); color: #e6eef3; font-weight:700; }
    .stTabs [aria-selected="true"] { background: linear-gradient(90deg,#ff3c8c,#6c63ff) !important; color: white !important; }

    .sidebar-title { font-size:20px; font-weight:800; text-align:center; margin-bottom:8px; background: linear-gradient(90deg,#ff3c8c,#6c63ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
    .sidebar-box { background: rgba(255,255,255,0.02); padding:12px; border-radius:12px; margin-bottom:8px; border:1px solid rgba(255,255,255,0.04); }
    .sidebar-item { padding:8px 12px; border-radius:10px; transition:0.15s; cursor:pointer; font-weight:600; color:#e6eef3; }
    .sidebar-item:hover { transform: translateX(6px); background: linear-gradient(90deg,#ff3c8c,#6c63ff); }

    .fact-card { padding:12px; border-radius:10px; margin-bottom:10px; background: rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.03); }

    .loader-container { width:100%; text-align:center; padding:14px 0; }
    .ai-loader { width:72px; height:72px; border-radius:50%; border-top:5px solid #ff3c8c; border-right:5px solid transparent; animation:spin 1s linear infinite; margin:auto; box-shadow:0 0 18px #ff3c8c55; }
    @keyframes spin { 0%{transform:rotate(0deg);} 100%{transform:rotate(360deg);} }
    .pulse-loader { display:flex; justify-content:center; margin-top:10px; }
    .pulse-loader div { width:12px; height:12px; margin:4px; border-radius:50%; background:linear-gradient(90deg,#ff3c8c,#6c63ff); animation:pulse 0.6s infinite alternate; }
    .pulse-loader div:nth-child(2) { animation-delay: 0.18s; }
    .pulse-loader div:nth-child(3) { animation-delay: 0.36s; }
    @keyframes pulse { from { transform:scale(1); opacity:0.6;} to { transform:scale(1.6); opacity:1;} }
    .bar-loader { width:220px; height:10px; background: rgba(255,255,255,0.06); border-radius:12px; margin:auto; overflow:hidden; box-shadow:0 0 12px rgba(108,99,255,0.12); }
    .bar-inner { width:40%; height:100%; background: linear-gradient(90deg,#6c63ff,#ff3c8c); animation:loading 1.2s infinite; }
    @keyframes loading { 0% { margin-left:-40%; } 100% { margin-left:140%; } }

    .muted { color:#b8c6d6; font-size:0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Sidebar content
# ---------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-box'><h2 class='sidebar-title'>⚡ AI Fact-Check</h2></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-item'>🏠 Home</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-item'>📰 Scraper</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-item'>🤖 Model Showdown</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-item'>🔎 Fact Check</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-item'>📊 Accuracy</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Text cleaning helpers
# ---------------------------
def clean_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    try:
        s = fix_text(s)
    except Exception:
        pass
    return " ".join(str(s).split()).strip()

# ---------------------------
# Load spaCy model safely
# ---------------------------
@st.cache_resource
def load_spacy_model():
    try:
        nlp_local = spacy.load("en_core_web_sm")
        return nlp_local
    except Exception as e:
        st.error("SpaCy model 'en_core_web_sm' not installed in the environment. Add the model wheel to requirements or run: python -m spacy download en_core_web_sm locally.")
        raise e

try:
    NLP = load_spacy_model()
except Exception:
    st.stop()

stop_words = STOP_WORDS

# ---------------------------
# Politifact scraping
# ---------------------------
@st.cache_data(ttl=60*60*24)
def scrape_politifact(start_date: pd.Timestamp, end_date: pd.Timestamp, max_pages: int = 50) -> pd.DataFrame:
    base = "https://www.politifact.com/factchecks/list/"
    rows = []
    url = base
    page = 0
    visited = set()
    while url and page < max_pages:
        page += 1
        try:
            r = requests.get(url, timeout=12)
            r.raise_for_status()
        except Exception as e:
            logger.warning(f"fetch fail {url}: {e}")
            break
        soup = BeautifulSoup(r.text, "html.parser")
        items = soup.find_all("li", class_="o-listicle__item")
        if not items:
            break
        stop_page = False
        for card in items:
            date_div = card.find("div", class_="m-statement__desc")
            date_text = date_div.get_text(" ", strip=True) if date_div else ""
            claim_date = None
            if date_text:
                m = re.search(r"stated on ([A-Za-z]+\s+\d{1,2},\s+\d{4})", date_text)
                if m:
                    try:
                        claim_date = pd.to_datetime(m.group(1), format="%B %d, %Y")
                    except Exception:
                        claim_date = pd.to_datetime(m.group(1), errors='coerce')
            if claim_date is None:
                continue
            if claim_date < start_date:
                stop_page = True
                break
            if not (start_date <= claim_date <= end_date):
                continue
            stmt = None
            stmt_block = card.find("div", class_="m-statement__quote")
            if stmt_block:
                a = stmt_block.find("a", href=True)
                if a:
                    stmt = clean_text(a.get_text(" ", strip=True))
            label = None
            img = card.find("img", alt=True)
            if img and 'alt' in img.attrs:
                label = clean_text(img['alt'].title())
            source = None
            src_a = card.find("a", class_="m-statement__name")
            if src_a:
                source = clean_text(src_a.get_text(" ", strip=True))
            rows.append({"date": claim_date.strftime("%Y-%m-%d"), "statement": stmt, "label": label, "source": source})
        if stop_page:
            break
        next_link = soup.find("a", class_="c-button c-button--hollow", string=re.compile(r"Next", re.I))
        if next_link and next_link.get("href"):
            url = urljoin(base, next_link['href'])
        else:
            break
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["statement", "label"])
    if not df.empty:
        df.to_csv("politifact_data.csv", index=False)
    return df

# ---------------------------
# Feature helpers
# ---------------------------
def semantic_features(texts: List[str]) -> pd.DataFrame:
    rows = []
    for t in texts:
        b = TextBlob(str(t))
        rows.append([b.sentiment.polarity, b.sentiment.subjectivity])
    return pd.DataFrame(rows, columns=["polarity", "subjectivity"])

def map_label_to_binary(label):
    REAL_LABELS = ["True", "No Flip", "Mostly True", "Half Flip", "Half True"]
    FAKE_LABELS = ["False", "Barely True", "Pants On Fire", "Full Flop"]
    if pd.isna(label):
        return np.nan
    l = str(label).strip()
    if l in REAL_LABELS:
        return 1
    if l in FAKE_LABELS:
        return 0
    low = l.lower()
    if "true" in low and "mostly" not in low and "half" not in low:
        return 1
    if "false" in low or "pants" in low or "fire" in low:
        return 0
    return np.nan

# ---------------------------
# Train accuracy model (TF-IDF + Logistic)
# ---------------------------
@st.cache_resource
def train_factcheck_accuracy_model(return_model=False):
    if not os.path.exists("politifact_data.csv"):
        raise FileNotFoundError("politifact_data.csv missing. Use Scraper tab or upload dataset.")
    df = pd.read_csv("politifact_data.csv")
    df["target"] = df["label"].apply(map_label_to_binary)
    df = df.dropna(subset=["target"])
    df = df[df["statement"].astype(str).str.len() > 10]
    X_texts = df["statement"].astype(str).tolist()
    y = df["target"].astype(int).values
    vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1,2))
    X = vectorizer.fit_transform(X_texts)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0))
    }
    if return_model:
        return model, vectorizer, metrics
    return metrics

def build_predictor():
    model, vectorizer, _ = train_factcheck_accuracy_model(return_model=True)
    def predict(statement: str):
        v = vectorizer.transform([statement])
        p = model.predict(v)[0]
        return "TRUE" if int(p) == 1 else "FALSE"
    return predict

# ---------------------------
# Google FactCheck API
# ---------------------------
def get_fact_check_results(query: str, api_key: str = API_KEY):
    if not api_key:
        return []
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "key": api_key}
    try:
        resp = requests.get(url, params=params, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        out = []
        for claim in data.get("claims", []):
            for review in claim.get("claimReview", []):
                out.append({
                    "text": claim.get("text", ""),
                    "publisher": review.get("publisher", {}).get("name", "Unknown"),
                    "title": review.get("title", ""),
                    "rating": review.get("textualRating", ""),
                    "url": review.get("url", "")
                })
        return out
    except Exception as e:
        logger.warning(f"FactCheck API error: {e}")
        return []

# ---------------------------
# UI: Tabs
# ---------------------------
st.markdown("<div class='app-title'>🚀 AI Fact-Check — Premium Dashboard</div>", unsafe_allow_html=True)
tabs = st.tabs(["Home", "Scraper", "Model Showdown", "Fact Check", "Fact-Check Accuracy"])

# HOME
with tabs[0]:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='margin:0; font-size:22px; font-weight:800; background:linear-gradient(90deg,#6c63ff,#ff3c8c); -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>Welcome</h2>", unsafe_allow_html=True)
    st.write("This portal scrapes Politifact, trains models to classify statements as TRUE/FALSE, evaluates models, and cross-checks claims using Google's FactCheck Tools.")
    st.markdown("<hr/>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.markdown("<div class='glass-card'><h3 style='margin:0'>📰 Scraped Claims</h3><h2 style='margin:0'>—</h2><p class='muted'>Use Scraper tab to fetch data</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='glass-card'><h3 style='margin:0'>🤖 Model Count</h3><h2 style='margin:0'>4</h2><p class='muted'>NaiveBayes, Logistic, SVM, DecisionTree</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='glass-card'><h3 style='margin:0'>📊 Typical Accuracy</h3><h2 style='margin:0'>~70-88%</h2><p class='muted'>Varies by features & dataset</p></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# SCRAPER
with tabs[1]:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.header("Politifact Scraper")
    st.write("Scrape Politifact claims by date range and save to `politifact_data.csv`.")
    min_date = pd.to_datetime("2007-01-01")
    max_date = pd.to_datetime("today").normalize()
    start_date = st.date_input("Start date", min_value=min_date, max_value=max_date, value=pd.to_datetime("2023-01-01"))
    end_date = st.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)
    if st.button("Scrape Politifact ⛏️"):
        if start_date > end_date:
            st.error("Start date must be <= end date.")
        else:
            st.markdown("<div class='loader-container'><div class='ai-loader'></div><div style='margin-top:8px; font-weight:700; background:linear-gradient(90deg,#ff3c8c,#6c63ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>⚡ Scraping Politifact...</div></div>", unsafe_allow_html=True)
            try:
                df_scraped = scrape_politifact(pd.to_datetime(start_date), pd.to_datetime(end_date), max_pages=80)
                if df_scraped.empty:
                    st.warning("No claims scraped. Site structure may have changed or date range had no matches.")
                else:
                    st.success(f"Scraped {len(df_scraped)} claims and saved to politifact_data.csv")
                    st.dataframe(df_scraped.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Scraping failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# MODEL SHOWDOWN
with tabs[2]:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.header("Model Showdown (Quick Compare)")
    st.write("Train & compare classical models using TF-IDF features. Requires politifact_data.csv.")
    if not os.path.exists("politifact_data.csv"):
        st.info("No local politifact_data.csv found. Scrape in Scraper tab first.")
    else:
        if st.button("Run quick model comparison"):
            st.markdown("<div class='loader-container'><div class='pulse-loader'><div></div><div></div><div></div></div><div style='margin-top:8px; font-weight:700; background:linear-gradient(90deg,#6c63ff,#ff3c8c); -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>⚡ Training & Evaluating...</div></div>", unsafe_allow_html=True)
            try:
                df_local = pd.read_csv("politifact_data.csv")
                df_local["target"] = df_local["label"].apply(map_label_to_binary)
                df_local = df_local.dropna(subset=["target"])
                df_local = df_local[df_local["statement"].astype(str).str.len() > 10]
                X_texts = df_local["statement"].astype(str).tolist()
                y = df_local["target"].astype(int).values
                vect = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
                X = vect.fit_transform(X_texts)
                models = {
                    "Naive Bayes": MultinomialNB(),
                    "Logistic Regression": LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced'),
                    "SVM (linear)": SVC(kernel='linear', C=0.5, class_weight='balanced'),
                    "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42)
                }
                results = []
                skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
                for name, clf in models.items():
                    accs, f1s, precs, recs = [], [], [], []
                    for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
                        X_train = X[train_idx]; X_test = X[test_idx]
                        y_train = y[train_idx]; y_test = y[test_idx]
                        try:
                            if name == "Naive Bayes":
                                clf.fit(X_train, y_train)
                                preds = clf.predict(X_test)
                            else:
                                pipeline = ImbPipeline([("smote", SMOTE(random_state=42, k_neighbors=3)), ("clf", clf)])
                                pipeline.fit(X_train, y_train)
                                preds = pipeline.predict(X_test)
                        except Exception as e:
                            logger.warning(f"Model {name} fold error: {e}")
                            preds = np.zeros_like(y_test)
                        accs.append(accuracy_score(y_test, preds))
                        f1s.append(f1_score(y_test, preds, zero_division=0))
                        precs.append(precision_score(y_test, preds, zero_division=0))
                        recs.append(recall_score(y_test, preds, zero_division=0))
                    results.append({
                        "Model": name,
                        "Accuracy": np.mean(accs)*100,
                        "F1": np.mean(f1s),
                        "Precision": np.mean(precs),
                        "Recall": np.mean(recs)
                    })
                res_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
                st.dataframe(res_df, use_container_width=True)
            except Exception as e:
                st.error(f"Model showdown failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# FACT CHECK (Google API)
with tabs[3]:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.header("Cross-Platform Fact Check (Google FactCheck Tools)")
    st.write("Query Google FactCheck Tools API. Configure your API key in Streamlit Secrets to avoid usage limits.")
