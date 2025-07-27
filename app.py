# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import requests
from io import BytesIO
import re

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

# ───── PAGE SETUP & CSS ─────
st.set_page_config("Restaurant Finder", "🍽️", layout="wide")
st.markdown("""
<style>
.block-container {padding:2rem;}
[data-testid="stSidebar"] {background:#f7f7f9;}
.badge {display:inline-block; background:#ffb300; color:#fff; border-radius:4px; padding:4px 8px; margin:4px; font-size:0.85rem;}
.review-snippet {margin:4px 0; padding-left:1rem; border-left:3px solid #43a047;}
.address {font-style:italic; color:#555; margin-bottom:8px;}
</style>
""", unsafe_allow_html=True)

# ───── DATA LOADING ─────
DATA_URL = "https://drive.google.com/uc?export=download&id=1iRWeGaDybybQ2eiTCyEgyXDYbH5FpFup"

@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(BytesIO(resp.content))

df_raw = load_data(DATA_URL)

# ───── MODEL TRAINING ─────
@st.cache_resource
def train_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["google_rating","price","popularity","sentiment"])
    num_feats = ["price","popularity","sentiment"]
    cat_feats = ["category"]
    X = df[num_feats + cat_feats]
    y = df["google_rating"]

    prep = ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
    ])

    models = {
        "RF": RandomForestRegressor(n_estimators=150, random_state=42),
        "GB": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    }

    best_rmse, best_pipe = np.inf, None
    for model in models.values():
        pipe = Pipeline([("prep", prep), ("reg", model)])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        rmse = mean_squared_error(y, preds) ** 0.5
        if rmse < best_rmse:
            best_rmse, best_pipe = rmse, pipe

    best_pipe.fit(X, y)
    df["predicted_rating"] = best_pipe.predict(X).round(1)
    return df

df = train_model(df_raw)

# ───── FILTERS ─────
st.sidebar.header("Filters")

# City
if "city" in df.columns:
    cities = ["All"] + sorted(df["city"].dropna().unique())
    city = st.sidebar.selectbox("City", cities)
    if city != "All":
        df = df[df["city"] == city]

# ZIP code
if "postal_code" in df.columns:
    zips = ["All"] + sorted(df["postal_code"].dropna().unique())
    zp = st.sidebar.selectbox("ZIP code", zips)
    if zp != "All":
        df = df[df["postal_code"] == zp]

# Category
if "category" in df.columns:
    cats = sorted(df["category"].dropna().unique())
    sel = st.sidebar.multiselect("Category", cats)
    if sel:
        df = df[df["category"].isin(sel)]

# Price level 1–10
pmin, pmax = 1, 10
pr = st.sidebar.slider("Price level", pmin, pmax, (pmin, pmax))
df = df[df["price"].between(pr[0], pr[1])]

# ───── SELECT TOP 5 ─────
df = df.sort_values("predicted_rating", ascending=False)
top5 = df.head(5).reset_index(drop=True)

# ───── MAIN PAGE ─────
st.title("🍴 Top 5 Restaurants by Predicted Rating")
c1, c2, c3 = st.columns(3)
c1.metric("Matches", len(df))
c2.metric("Avg Predicted Rating", f"{df['predicted_rating'].mean():.2f}")
c3.metric("Avg Sentiment", f"{df['sentiment'].mean():.2f}")

# User selects one restaurant
names = [""] + list(top5["name"])
sel = st.selectbox("Select a restaurant to inspect", names)
if not sel:
    st.info("Please select a restaurant above.")
    st.stop()

r = top5[top5["name"] == sel].iloc[0]

# ───── FORMATTED ADDRESS ─────
if "formatted_address" in r:
    st.markdown(f"<div class='address'>📍 {r['formatted_address']}</div>", unsafe_allow_html=True)

# ───── MAP ─────
if {"latitude","longitude"}.issubset(r.index):
    view = pdk.ViewState(latitude=r["latitude"], longitude=r["longitude"], zoom=14)
    color = [
        int(255*(1-(r["predicted_rating"]-1)/4)),
        int(120+135*(r["predicted_rating"]-1)/4),
        200, 180
    ]
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame([r]),
        get_position='[longitude, latitude]',
        get_fill_color=color,
        get_radius=100
    )
    st.pydeck_chart(pdk.Deck(initial_view_state=view, layers=[layer]))
else:
    st.error("Location data missing.")

# ───── KEY PHRASE EXTRACTION ─────
st.subheader(f"{sel} — Key Phrases from Reviews")
raw_text = r.get("combined_reviews", "")
if raw_text:
    # Split raw_text into individual review snippets
    docs = [s.strip() for s in re.split(r'\|\||\n', raw_text) if s.strip()]
    # TF-IDF with n-grams (1- and 2-word phrases)
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=50)
    X = vect.fit_transform(docs)
    scores = np.asarray(X.mean(axis=0)).ravel()
    terms = np.array(vect.get_feature_names_out())
    top_phrases = terms[np.argsort(scores)[::-1][:8]]
    badges = " ".join(f"<span class='badge'>{w}</span>" for w in top_phrases)
    st.markdown(badges, unsafe_allow_html=True)
else:
    st.info("No review text available.")

# ───── REVIEW SNIPPETS ─────
st.subheader("Sample Reviews")
if raw_text:
    snippets = docs[:3]  # first three processed reviews
    for snippet in snippets:
        st.markdown(f"<div class='review-snippet'>“{snippet}”</div>", unsafe_allow_html=True)
else:
    st.info("No reviews to display.")
