# 📊 Twitter Sentiment Analysis for Stock Market Forecasting

This project explores the relationship between financial sentiment from social media (tweets) and stock market performance, using deep learning (LSTM) to forecast stock prices based on sentiment trends.

---

## 🔍 Project Overview

The aim is to:

* Analyze sentiment in stock-related tweets using NLP (VADER + custom financial context)
* Merge tweet sentiment scores with historical stock data
* Engineer features like daily returns and lagged sentiment values
* Train an LSTM model to predict future stock prices based on past sentiment and price trends

---

## 📂 Dataset Description

This project uses two key datasets:

* `stock_tweets.csv` – Contains stock-related tweets with timestamps and stock tickers
* `stock_yfinance_data.csv` – Historical daily stock data (Open, High, Low, Close, Volume)

**Processed Dataset (generated):**

* `processed_stock_sentiment_with_lags.csv` – Final dataset with:
  * Daily sentiment scores
  * Daily returns
  * Lagged sentiment features
  * Scaled features for model training

---

## 🛠️ Key Components

* **NLP & Sentiment Analysis:**  
  Using `nltk` and `SentimentIntensityAnalyzer (VADER)`, tweets are scored with adjustments for financial terms like *bullish*, *buy*, *loss*, and domain-specific terms like *Oculus*.

* **Feature Engineering:**  
  Lagged sentiment features (t-1, t-2, t-3) and daily returns are calculated to capture short-term market trends.

* **Time Series Modeling with LSTM:**  
  Built using `TensorFlow/Keras`, the LSTM learns sequential patterns from the sentiment and price data.

* **Visualization:**  
  Scaled comparison plots of sentiment trends vs. stock prices over time (daily and weekly).

---

## 🧪 Evaluation Metrics

* MAE (Mean Absolute Error)  
* RMSE (Root Mean Squared Error)  
* MAPE (Mean Absolute Percentage Error)

These metrics help assess the performance of the LSTM model in forecasting stock prices.

---

## 🚀 How to Run the Project

### ▶️ Run in Google Colab

**1. Open the Notebook**

* Go to Google Colab and upload the notebook file: `stock_sentiment_prediction.ipynb`

**2. Upload CSV Files to Colab Runtime**

* In the Colab sidebar:
  * Click the folder 📁 icon
  * Click "Upload"
  * Upload both:
    * `stock_tweets.csv`
    * `stock_yfinance_data.csv`

**3. Verify Upload Paths in Code**

Make sure this line exists and matches your upload:

<pre><code>tweets_df = pd.read_csv('/content/stock_tweets.csv')</code></pre>

---

## 📦 Installing Required Libraries

To run this project, install the following libraries:

* pandas  
* numpy  
* nltk  
* matplotlib  
* seaborn  
* scikit-learn  
* tensorflow  

Run this command in a Colab cell or your terminal:

<pre><code>pip install pandas numpy nltk matplotlib seaborn scikit-learn tensorflow</code></pre>
