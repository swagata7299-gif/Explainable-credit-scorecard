import os, time, json, yfinance as yf, feedparser, pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

def fetch_prices(tickers, period="6mo", interval="1d"):
    frames = []
    for t in tickers:
        df = yf.download(t, period=period, interval=interval, auto_adjust=True, progress=False)
        df["ticker"] = t
        frames.append(df.reset_index())
        time.sleep(0.3)
    out = pd.concat(frames, ignore_index=True)
    out.rename(columns={"Date":"date"}, inplace=True)
    return out  # cols: date, Open, High, Low, Close, Volume, ticker

def fetch_fundamentals(ticker):
    tk = yf.Ticker(ticker)
    info = tk.fast_info.__dict__ if hasattr(tk, "fast_info") else {}
    return pd.DataFrame([info]).assign(ticker=ticker)

def fetch_rss(feed_url, limit=50):
    d = feedparser.parse(feed_url)
    items = []
    for e in d.entries[:limit]:
        title = getattr(e, "title", "")
        summary = BeautifulSoup(getattr(e, "summary", ""), "html.parser").get_text()
        link = getattr(e, "link", "")
        published = getattr(e, "published", "") or getattr(e, "updated", "")
        items.append({"title": title, "summary": summary, "link": link, "published": published})
    return pd.DataFrame(items)

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

if __name__ == "__main__":
    TICKERS = ["AAPL","MSFT"]  # demo: 2 structured sources (prices+fundamentals), 1 unstructured (news RSS)
    prices = fetch_prices(TICKERS)
    save_csv(prices, "data/raw/prices.csv")
    fundamentals = pd.concat([fetch_fundamentals(t) for t in TICKERS], ignore_index=True)
    save_csv(fundamentals, "data/raw/fundamentals.csv")
    # Unstructured: Yahoo Finance news RSS (free/public)
    news = pd.concat([fetch_rss(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}") for t in TICKERS], ignore_index=True)
    save_csv(news, "data/raw/news.csv")
    print("Ingestion complete.")
