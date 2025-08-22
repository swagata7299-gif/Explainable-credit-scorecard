


📊 Explainable Credit Scorecard

An **AI-powered, real-time creditworthiness scoring system** that integrates structured financial data (prices, fundamentals) with unstructured signals (news sentiment & events). Built with **interpretable ML models + SHAP explainability**, and delivered through an interactive Streamlit dashboard.



🚀 Features

High-Throughput Data Ingestion

  * Structured: Yahoo Finance prices & fundamentals (yfinance API)
  * Unstructured: Real-time RSS news headlines with NLP & sentiment analysis
  Adaptive Scoring Engine**

  * Incremental retraining (`SGDClassifier`)
  * Predicts credit score on a 0–100 scale
* Explainability Layer**

  * SHAP waterfall plot (single sample explanation)
  * SHAP summary plot (global feature importance)
  * Plain-language risk signals extracted from news
* Interactive Dashboard

  * Streamlit app for analysts
  * Ticker selection, score trends, feature importance visualizations
  * Event-driven explanations (e.g., “Debt restructuring → Score decrease”)
* Deployment Ready

  * Dockerized for reproducibility
  * GitHub Actions workflow for automated data refresh & retraining

---

 🏗️ System Architecture

![Architecture](docs/architecture.png)

1. Data Ingestion

   * Fetch prices, fundamentals, and news headlines
   * Cache data in `data/raw/`

2. Feature Engineering

   * Compute returns, volatility, sentiment/risk signals
   * Merge structured + unstructured → `data/processed/`

3. Scoring Engine

   * Train interpretable ML model (SGDClassifier)
   * Incremental retraining for near real-time updates

4. Explainability Layer

   * SHAP plots + news-driven plain-text reasoning
     
5. Analyst Dashboard

   * Streamlit interface with trends, filters, and alerts



📂 Repository Structure


explainable-credit-scorecard/
│── README.md
│── requirements.txt
│── Dockerfile
│── app.py                  # CLI runner
│── dashboard.py            # Streamlit analyst UI
│── src/
│   ├── ingestion.py        # structured + unstructured collectors
│   ├── features.py         # feature engineering
│   ├── events_nlp.py       # NLP + risk signal extraction
│   ├── scoring_engine.py   # model training, retraining, SHAP
│   └── utils.py
│── data/
│   ├── raw/
│   └── processed/
│── outputs/
│   ├── shap_waterfall.png
│   └── shap_summary.png
│── docs/
│   └── architecture.png


🛠️ Installation & Run
1️⃣ Local Setup

bash
git clone https://github.com/<your-username>/explainable-credit-scorecard.git
cd explainable-credit-scorecard
pip install -r requirements.txt


 2️⃣ Run Streamlit Dashboard

bash
streamlit run dashboard.py


3️⃣ Docker (Optional)

bash
docker build -t credit-scorecard .
docker run -p 8501:8501 credit-scorecard




📊 Outputs

Example explainability plots:

| Local Explanation (Waterfall)            | Global Importance (Summary)          |
| ---------------------------------------- | ------------------------------------ |
| ![Waterfall](outputs/shap_waterfall.png) | ![Summary](outputs/shap_summary.png) |


 🎯 Evaluation Alignment

* Data Engineering (20%)→ Robust ingestion, error handling, caching
* Model Accuracy & Explainability (30%)→ SHAP, interpretable & retrainable model
* Unstructured Integration (12.5%) → NLP-driven risk signals from news
* UX & Dashboard (15%) → Analyst-friendly Streamlit interface
* Deployment & Ops (10%) → Docker + GitHub Actions refresh
* Innovation (12.5%)→ Event-to-risk factor mapping




---

👉 Do you want me to also draft the **presentation slides outline** (titles + bullets per slide) so your PPT is ready in 10 minutes?

