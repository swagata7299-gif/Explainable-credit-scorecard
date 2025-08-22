import re, pandas as pd, spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_sm")
sid = SentimentIntensityAnalyzer()

RISK_RULES = {
    r"debt (restructuring|refinanc)": +0.25,
    r"downgrade|rating cut": +0.2,
    r"default|distress": +0.3,
    r"layoff|downsizing": +0.15,
    r"guidance (cut|lower)": +0.15,
    r"miss(es)? (earnings|revenue)": +0.15,
    r"investigation|probe|fraud": +0.35,
    r"bankruptcy|chapter 11": +0.5,
    r"beats? (earnings|revenue)": -0.1,
    r"guidance (raise|higher)": -0.1,
    r"buyback|dividend (increase|hike)": -0.08
}

def score_event(text):
    text_l = text.lower()
    score = 0.0
    for pat, w in RISK_RULES.items():
        if re.search(pat, text_l):
            score += w
    sent = sid.polarity_scores(text_l)["compound"]
    # map sentiment to risk: negative → +risk, positive → -risk
    score += -0.2*sent
    return max(min(score, 1.0), -1.0)

def enrich_news(df_news, tickers):
    # crude ticker/entity matching by title text
    rows = []
    for _, r in df_news.iterrows():
        txt = f"{r.get('title','')} {r.get('summary','')}"
        ev = score_event(txt)
        ent = [e.text for e in nlp(txt).ents if e.label_ in {"ORG","PRODUCT","GPE"}]
        rows.append({**r, "risk_signal": ev, "entities": ",".join(ent)})
    out = pd.DataFrame(rows)
    # optional: filter per ticker keyword
    outs = []
    for t in tickers:
        mask = out["title"].str.contains(t, case=False, na=False) | out["summary"].str.contains(t, case=False, na=False)
        dft = out[mask].copy()
        dft["ticker"] = t
        outs.append(dft)
    return pd.concat(outs, ignore_index=True) if outs else out
