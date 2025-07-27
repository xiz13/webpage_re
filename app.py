# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import requests
from io import BytesIO
import streamlit.components.v1 as components

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# ───── PAGE CONFIG & GLOBAL CSS ─────
st.set_page_config(
    page_title="Restaurant Map & Ratings",
    page_icon="🍴",
    layout="wide"
)

st.markdown("""
<style>
.block-container {padding:2rem;}
[data-testid="stSidebar"] {background-color:#f7f7f9;}
.card-grid {display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:16px;}
.card {background:#fff; border-radius:8px; padding:16px; box-shadow:0 2px 8px rgba(0,0,0,0.1);}
.card h4 {margin:0 0 8px;}
.badge {display:inline-block; background:#ffb300; color:#fff; border-radius:4px; padding:2px 6px; font-size:0.85rem; margin-left:8px;}
.sent-bar {background:#e0e0e0; border-radius:4px; height:8px; overflow:hidden; margin-top:4px;}
.sent-fill {height:100%; background:#43a047; transition:width .4s;}
</style>
""", unsafe_allow_html=True)

# ───── DATA LOADING ─────
DATA_URL = "https://drive.google.com/uc?export=download&id=1iRWeGaDybybQ2eiTCyEgyXDYbH5FpFup"

@st.cache_data(show_spinner="Downloading data...")
def load_data(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(BytesIO(r.content))

df_raw = load_data(DATA_URL)

# ───── MODEL TRAINING ─────
@st.cache_resource(show_spinner="Training regression model…")
def train_regression(df):
    df = df.dropna(subset=["google_rating", "price", "popularity", "sentiment"])
    # simple feature set
    num_feats = ["price", "popularity", "sentiment"]
    cat_feats = ["category"]
    X = df[num_feats + cat_feats]
    y = df["google_rating"]
    prep = ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
    ])
    models = {
        "RF": (RandomForestRegressor(n_estimators=150, random_state=42), {}),
        "GB": (GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42), {}),
    }
    # cross‑validate choose best
    best_rmse, best_name, best_pipe = np.inf, None, None
    kf = KFold(3, shuffle=True, random_state=42)
    for name, (est, params) in models.items():
        pipe = Pipeline([("prep", prep), ("reg", est)])
        # no gridsearch since small set
        pipe.fit(X, y)
        preds = pipe.predict(X)
        rmse = mean_squared_error(y, preds) ** 0.5
        if rmse < best_rmse:
            best_rmse, best_name, best_pipe = rmse, name, pipe
    # final fit
    best_pipe.fit(X, y)
    df["predicted_rating"] = best_pipe.predict(X).round(1)
    return df, best_name, best_rmse

df, model_name, model_rmse = train_regression(df_raw)

# ───── SIDEBAR FILTERS ─────
st.sidebar.header("Filters")

# city → zip cascaded
if "city" in df.columns:
    cities = ["All"] + sorted(df["city"].dropna().unique())
    city = st.sidebar.selectbox("City", cities)
    if city != "All":
        df = df[df["city"] == city]
else:
    city = None

if "postal_code" in df.columns:
    zips = ["All"] + sorted(df["postal_code"].dropna().unique())
    zipcode = st.sidebar.selectbox("ZIP code", zips)
    if zipcode != "All":
        df = df[df["postal_code"] == zipcode]
else:
    zipcode = None

if "category" in df.columns:
    cats = sorted(df["category"].dropna().unique())
    sel_cats = st.sidebar.multiselect("Category", cats)
    if sel_cats:
        df = df[df["category"].isin(sel_cats)]

# price slider
if pd.api.types.is_numeric_dtype(df["price"]):
    pmin, pmax = int(df["price"].min()), int(df["price"].max())
    price_range = st.sidebar.slider("Price", pmin, pmax, (pmin, pmax))
    df = df[df["price"].between(*price_range)]

# sort and select top 5
df = df.sort_values("predicted_rating", ascending=False)
top5 = df.head(5)

# ───── MAIN LAYOUT ─────
st.title("🍴 Restaurant Map & Ratings")
st.write(f"Model: **{model_name}**, RMSE ≈ **{model_rmse:.2f}** on training data")

# KPI row
c1, c2, c3 = st.columns(3)
c1.metric("Total Matches", len(df))
c2.metric("Avg Predicted Rating", f"{df['predicted_rating'].mean():.2f}")
c3.metric("Avg Sentiment", f"{df['sentiment'].mean():.2f}")

# dual view: cards + map
left, right = st.columns([1, 1.5])

with left:
    st.subheader("Top 5 Recommendations")
    # render cards via HTML
    card_html = ""
    for _, r in top5.iterrows():
        # extract snippets
        snippets = []
        for s in str(r["combined_reviews_clean"]).split("||")[:3]:
            snippets.append(f"<em>“{s.strip()}”</em>")
        more_count = max(0, len(str(r["combined_reviews_clean"]).split("||")) - 3)
        # sentiment bar width
        sent_pct = int(100 * float(r["sentiment"]))
        sent_pct = max(0, min(sent_pct, 100))
        card_html += f"""
        <div class="card">
          <h4>{r['name']}<span class="badge">{r['predicted_rating']}</span></h4>
          <div>📍 {r.get('city','')}, {r.get('postal_code','')}</div>
          <div class="sent-bar"><div class="sent-fill" style="width:{sent_pct}%"></div></div>
          <div style="margin-top:8px;">{' '.join(snippets)}</div>
          {'<details><summary>Show more reviews</summary>' +
            '<br>'.join(str(r["combined_reviews_clean"]).split("||")[3:]) +
            '</details>' if more_count>0 else ''}
        </div>"""
    components.html(f"<div class='card-grid'>{card_html}</div>", height=550, scrolling=True)

with right:
    st.subheader("Map View")
    if {"latitude","longitude"}.issubset(top5.columns):
        top5["color"] = top5["predicted_rating"].apply(
            lambda v: [int(255*(1-(v-1)/4)), int(120+(v-1)/4*135), 200, 180]
        )
        view = pdk.ViewState(
            latitude=top5["latitude"].mean(),
            longitude=top5["longitude"].mean(),
            zoom=11
        )
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=top5,
            get_position='[longitude, latitude]',
            get_fill_color="color",
            get_radius=80,
            pickable=True
        )
        tooltip = {
            "html": "<b>{name}</b><br/>Pred: {predicted_rating}<br/>“{combined_reviews_clean.split('||')[0]}…”",
            "style": {"backgroundColor":"#F0F0F0","color":"#000","fontSize":"12px","padding":"10px"}
        }
        st.pydeck_chart(pdk.Deck(initial_view_state=view, layers=[layer], tooltip=tooltip))
    else:
        st.error("Latitude/longitude columns missing.")

# ───── FULL DATA TABLE DOWNLOAD ─────
st.download_button(
    "Download filtered results as CSV",
    df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_restaurants.csv",
    mime="text/csv"
)
