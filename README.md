# Nikkei 225 Market Trend Prediction
## Multimodal approach using BERT Sentiment Analysis & Financial Indicators

### 📌 Project Overview
This project explores the relationship between financial news sentiment and the daily movement of the Nikkei 225 index. The goal was to build a classification model that predicts whether the market will close "Up" or "Down" based on a combination of Natural Language Processing (NLP) and quantitative technical indicators.

### 🛠 Tech Stack
*   **Language:** Python 3.x
*   **NLP:** BERT (Bidirectional Encoder Representations from Transformers) via `HuggingFace Transformers`
*   **Data Source:** `yfinance` (Nikkei 225 historical prices), CSV-based news headlines
*   **Analysis:** Scikit-learn, Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn

### 📊 Methodology
1.  **NLP Pipeline:** Used a pre-trained **Japanese-finetuned BERT model** (`koheiduck/bert-japanese-finetuned-sentiment`) to extract sentiment scores from financial news headlines.
2.  **Feature Engineering:**
    *   **RSI (Relative Strength Index):** To identify overbought/oversold conditions.
    *   **SMA Ratio (Simple Moving Average):** 10-day vs 50-day ratio to capture trend momentum.
    *   **ATR (Average True Range):** Normalized to account for price volatility.
3.  **Model:** A Logistic Regression classifier with **balanced class weights** was used as a baseline to handle market directionality.

### 📈 Key Results & Analysis
*   **Baseline Accuracy:** ~58% across multiple cross-validation folds.
*   **Discovery:** The model showed a higher precision for "Up" trends but struggled during periods of extreme volatility.
*   **Post-Mortem:** Analysis of the results suggests that a "naive" sentiment approach (calculating a daily score without temporal weighting) may lag behind market reactions. The Efficient Market Hypothesis (EMH) suggests that public news is priced in rapidly, requiring more granular (intra-day) data for higher accuracy.

### 🚀 Future Improvements
*   **Temporal Weighting:** Implement Time-Series models (LSTM or GRU) to capture the decay of news impact over time.
*   **Alternative Text:** Expand beyond headlines to full-text financial reports using TF-IDF or specific industry lexicons.
*   **Hyperparameter Tuning:** Explore non-linear kernels using XGBoost or Random Forest to capture complex market interactions.

---
*Developed as part of my B.S. in Mathematical Sciences studies at Meiji University.*
