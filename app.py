# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import requests
from io import BytesIO
import streamlit.components.v1 as components
import re
from collections import Counter

# sklearn imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# ───── PAGE SETUP & CSS ─────
st.set_page_config("Restaurant Finder", "🍽️", layout="wide")
st.markdown("""
<style>
.block-container {padding:2rem;}
[data-testid="stSidebar"] {background:#f7f7f9;}
.card {background:#fff; border-radius:8px; padding:16px; box-shadow:0 2px 8px rgba(0,0,0,0.1); margin-bottom:12px;}
.keywords {font-size:0.9rem; color:#333; margin-bottom:8px;}
.review-expander summary {font-weight:600; cursor:pointer;}
</style>
""", unsafe_allow_html=True)

# ───── DATA LOAD ─────
DATA_URL = "https://drive.google.com/uc?export=download&id=1iRWeGaDybybQ2eiTCyEgyXDYbH5FpFup"

@st.cache_data
def load_data(url):
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(BytesIO(resp.content))

df_raw = load_data(DATA_URL)

# ───── MODEL ─────
@st.cache_resource
def train_model(df):
    # filter necessary columns
    df = df.dropna(subset=["google_rating", "price", "popularity", "sentiment"])
    num_feats = ["price", "popularity", "sentiment"]
    cat_feats = ["category"]
    X = df[num_feats + cat_feats]
    y = df["google_rating"]

    # preprocessing pipeline
    prep = ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats)
    ])

    # define models
    models = {
        "RF": RandomForestRegressor(n_estimators=150, random_state=42),
        "GB": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    }

    # select best via RMSE on training data
    best_rmse, best_pipe = np.inf, None
    for name, model in models.items():
        pipe = Pipeline([("prep", prep), ("reg", model)])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        rmse = mean_squared_error(y, preds) ** 0.5
        if rmse < best_rmse:
            best_rmse, best_pipe = rmse, pipe

    # final fit & attach predictions
    best_pipe.fit(X, y)
    df["predicted_rating"] = best_pipe.predict(X).round(1)
    return df

df = train_model(df_raw)

# ───── FILTERS ─────
st.sidebar.header("Filters")

# City filter
if "city" in df:
    cities = ["All"] + sorted(df["city"].dropna().unique())
    city = st.sidebar.selectbox("City", cities)
    if city != "All":
        df = df[df["city"] == city]

# ZIP code filter
if "postal_code" in df:
    zips = ["All"] + sorted(df["postal_code"].dropna().unique())
    zp = st.sidebar.selectbox("ZIP", zips)
    if zp != "All":
        df = df[df["postal_code"] == zp]

# Category filter
if "category" in df:
    cats = sorted(df["category"].dropna().unique())
    sel = st.sidebar.multiselect("Category", cats)
    if sel:
        df = df[df["category"].isin(sel)]

# Price slider (1–10)
pmin, pmax = 1, 10
price_range = st.sidebar.slider("Price level", pmin, pmax, (pmin, pmax))
df = df[df["price"].between(price_range[0], price_range[1])]

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
sel_name = st.selectbox("Choose a restaurant to inspect", names)
if not sel_name:
    st.info("Select a restaurant above to see details.")
    st.stop()

r = top5[top5["name"] == sel_name].iloc[0]

# ───── KEYWORD EXTRACTION ─────
clean_text = r.get("combined_reviews_clean", "")
words = re.findall(r"\b[a-zA-Z]{4,}\b", clean_text.lower())
common = [w for w, _ in Counter(words).most_common(8)]
st.subheader(f"{sel_name} — Predicted: {r['predicted_rating']} ⭐")
st.markdown("<div class='keywords'>Keywords: " + ", ".join(common) + "</div>", unsafe_allow_html=True)

# ───── MAP ─────
if {"latitude", "longitude"}.issubset(r.index):
    view = pdk.ViewState(latitude=r["latitude"], longitude=r["longitude"], zoom=14)
    color = [int(255*(1-(r['predicted_rating']-1)/4)),
             int(120+135*(r['predicted_rating']-1)/4), 200, 180]
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame([r]),
        get_position='[longitude, latitude]',
        get_fill_color=color,
        get_radius=100,
        pickable=True
    )
    st.pydeck_chart(pdk.Deck(initial_view_state=view, layers=[layer]))
else:
    st.error("Location data missing for this restaurant.")

# ───── REVIEWS EXPANDER ─────
st.subheader("Customer Reviews")
for i, snippet in enumerate(clean_text.split("||")):
    with st.expander(f"Review {i+1}", expanded=False):
        st.write(snippet.strip())
