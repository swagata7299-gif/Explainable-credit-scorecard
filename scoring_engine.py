import os, joblib, numpy as np, pandas as pd, shap, matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

MODEL_PATH = "outputs/model.joblib"

def train_or_load(df):
    X = df[["ret_1d","vol_z","risk_signal_mean","news_count","market_cap","year_high","year_low"]].fillna(0.0).values
    y = df["target"].values.astype(int)
    pipe = make_pipeline(StandardScaler(with_mean=False), SGDClassifier(loss="log_loss", max_iter=500, random_state=42))
    pipe.fit(X, y)
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    auc = roc_auc_score(y, pipe.predict_proba(X)[:,1])
    return pipe, auc

def incremental_update(pipe, df_new):
    Xn = df_new[["ret_1d","vol_z","risk_signal_mean","news_count","market_cap","year_high","year_low"]].fillna(0.0).values
    yn = df_new["target"].values.astype(int)
    # partial_fit for SGD
    pipe.named_steps["sgdclassifier"].partial_fit(
        pipe.named_steps["standardscaler"].transform(Xn), yn, classes=np.array([0,1])
    )
    joblib.dump(pipe, MODEL_PATH)
    return pipe

def explain(pipe, df_sample):
    Xs = df_sample[["ret_1d","vol_z","risk_signal_mean","news_count","market_cap","year_high","year_low"]].fillna(0.0)
    feat_names = list(Xs.columns)
    explainer = shap.Explainer(pipe.named_steps["sgdclassifier"], pipe.named_steps["standardscaler"].transform(Xs))
    sv = explainer(pipe.named_steps["standardscaler"].transform(Xs))
    # save visuals for first row
    shap.initjs()
    plt.figure()
    shap.plots.waterfall(sv[0], show=False)
    plt.savefig("outputs/shap_waterfall.png", dpi=300, bbox_inches="tight")
    plt.figure()
    shap.summary_plot(sv.values, Xs, plot_type="bar", show=False, feature_names=feat_names)
    plt.savefig("outputs/shap_summary.png", dpi=300, bbox_inches="tight")
    return sv
