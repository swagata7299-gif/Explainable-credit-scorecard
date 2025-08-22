import pandas as pd

def make_daily_features(prices_csv, fundamentals_csv, news_csv):
    px = pd.read_csv(prices_csv, parse_dates=["date"])
    fn = pd.read_csv(fundamentals_csv)
    nw = pd.read_csv(news_csv)
    # price features
    px = px.sort_values(["ticker","date"])
    px["ret_1d"] = px.groupby("ticker")["Close"].pct_change()
    px["vol_z"] = px.groupby("ticker")["Volume"].transform(lambda s: (s - s.mean())/(s.std()+1e-6))
    # fundamentals (demo: will be mostly static)
    fn_small = fn[["ticker","market_cap","year_high","year_low"]].fillna(0) if set(["market_cap","year_high","year_low"]).issubset(fn.columns) else fn.assign(market_cap=0, year_high=0, year_low=0)[["ticker","market_cap","year_high","year_low"]]
    # news signals
    nw["published"] = pd.to_datetime(nw["published"], errors="coerce")
    nw["date"] = nw["published"].dt.date
    news_sig = nw.groupby(["ticker","date"]).agg(risk_signal_mean=("risk_signal","mean"), news_count=("title","count")).reset_index()
    news_sig["date"] = pd.to_datetime(news_sig["date"])
    # join
    feat = px.merge(news_sig, on=["ticker","date"], how="left").fillna({"risk_signal_mean":0.0, "news_count":0})
    feat = feat.merge(fn_small, on="ticker", how="left")
    # label (demo): next-day return < -2% → “bad”; > +2% → “good”; else neutral
    feat["label"] = feat.groupby("ticker")["ret_1d"].shift(-1)
    feat = feat.dropna(subset=["label"])
    feat["target"] = (feat["label"] > 0.02).astype(int)  # binary for demo
    cols = ["ticker","date","Close","ret_1d","vol_z","risk_signal_mean","news_count","market_cap","year_high","year_low","target"]
    return feat[cols]
