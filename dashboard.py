import streamlit as st, pandas as pd, joblib
from src.ingestion import fetch_prices, fetch_fundamentals, fetch_rss
from src.events_nlp import enrich_news
from src.features import make_daily_features
from src.scoring_engine import train_or_load, explain

st.set_page_config(page_title="Explainable Credit Scorecard", layout="wide")
st.title("📊 Explainable Credit Scorecard")

tickers = st.multiselect("Select tickers", ["AAPL","MSFT","AMZN"], default=["AAPL","MSFT"])
if st.button("Run Pipeline"):
    prices = fetch_prices(tickers); prices.to_csv("data/raw/prices.csv", index=False)
    fundamentals = pd.concat([fetch_fundamentals(t) for t in tickers], ignore_index=True)
    fundamentals.to_csv("data/raw/fundamentals.csv", index=False)
    news = pd.concat([fetch_rss(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}") for t in tickers], ignore_index=True)
    news = enrich_news(news, tickers); news.to_csv("data/raw/news_scored.csv", index=False)

    feats = make_daily_features("data/raw/prices.csv","data/raw/fundamentals.csv","data/raw/news_scored.csv")
    feats.to_csv("data/processed/features.csv", index=False)

    model, auc = train_or_load(feats)
    st.success(f"Model trained. In-sample AUC: {auc:.3f}")

    # pick latest day per ticker
    latest = feats.sort_values("date").groupby("ticker").tail(1)
    explain(model, latest)

    st.image("outputs/shap_waterfall.png", caption="SHAP Waterfall (single sample)")
    st.image("outputs/shap_summary.png", caption="Global Feature Importance")

    # Score table
    latest["score"] = model.predict_proba(
        latest[["ret_1d","vol_z","risk_signal_mean","news_count","market_cap","year_high","year_low"]].fillna(0.0).values
    )[:,1].round(3)
    st.dataframe(latest[["ticker","date","score","risk_signal_mean","news_count"]].sort_values("score", ascending=False))
