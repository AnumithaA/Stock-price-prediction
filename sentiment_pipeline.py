# sentiment_pipeline.py
import os
import time
import numpy as np
import pandas as pd
from gnews import GNews
from datetime import date
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from datetime import datetime, timedelta

os.makedirs("data", exist_ok=True)

# ── FinBERT setup ──
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert   = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
finbert.eval()

def encode_headline(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    with torch.no_grad():
        outputs = finbert(**inputs, output_hidden_states=True)

    probs = F.softmax(outputs.logits, dim=-1)[0]
    score = (probs[2] - probs[0]).item()
    embedding = outputs.hidden_states[-1][0, 0, :].numpy()

    return score, embedding


# ── SIMPLIFIED QUERIES (faster) ──
QUERIES = {
    "AAPL": ["AAPL Apple stock news"],
    "TSLA": ["TSLA Tesla stock news"],
}

START_YEAR = 2019
END_YEAR   = date.today().year


# ── FIXED DATE RANGES ──
def month_ranges(start_year, end_year):
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):

            start_date = datetime(year, month, 1)

            if month == 12:
                next_month = datetime(year + 1, 1, 1)
            else:
                next_month = datetime(year, month + 1, 1)

            end_date = next_month - timedelta(days=1)

            yield (
                (start_date.year, start_date.month, start_date.day),
                (end_date.year, end_date.month, end_date.day)
            )


# ── NEWS FETCHING ──
def fetch_news_for_ticker(ticker):
    all_records = []

    for start, end in month_ranges(START_YEAR, END_YEAR):

        gn = GNews(language="en", country="US", max_results=50)
        gn.start_date = start
        gn.end_date = end

        for query in QUERIES[ticker]:

            print(f"Fetching {ticker} | {start} → {end} | {query}")

            try:
                results = gn.get_news(query)

                for item in results:
                    raw_date = item.get("published date", "")
                    headline = item.get("title", "").strip()

                    if not headline or not raw_date:
                        continue

                    try:
                        date = pd.to_datetime(raw_date).strftime("%Y-%m-%d")
                    except:
                        continue

                    all_records.append({
                        "date": date,
                        "headline": headline,
                        "ticker": ticker,
                    })

                time.sleep(0.8)

            except Exception as e:
                print(f"Warning: {ticker} {start}–{end} failed: {e}")
                time.sleep(2)

    df = pd.DataFrame(all_records).drop_duplicates(subset=["date", "headline"])

    print(f"{ticker}: {len(df)} headlines across {df['date'].nunique()} days")

    return df


# ── SENTIMENT AGGREGATION ──
def build_daily_sentiment(raw_df):
    records = []
    grouped = raw_df.groupby(["ticker", "date"])

    for (ticker, date), group in grouped:

        scores, embeddings = [], []

        for _, row in group.iterrows():
            try:
                score, emb = encode_headline(row["headline"])
                scores.append(score)
                embeddings.append(emb)
            except Exception as e:
                print(f"Encoding failed: {e}")

        if not scores:
            continue

        records.append({
            "ticker": ticker,
            "date": date,
            "sentiment_score": float(np.mean(scores)),
            "headline_count": len(scores),
            "embedding": np.mean(np.stack(embeddings), axis=0),
        })

    return pd.DataFrame(records)


# ── MAIN ──
if __name__ == "__main__":

    for ticker in ["AAPL", "TSLA"]:

        print(f"\n── Fetching news for {ticker} ──")

        raw = fetch_news_for_ticker(ticker)
        raw.to_csv(f"data/{ticker}_raw_news.csv", index=False)

        if raw.empty:
            print(f"No articles found for {ticker}")
            pd.DataFrame(columns=[
                "ticker", "date", "sentiment_score",
                "headline_count", "embedding"
            ]).to_pickle(f"data/{ticker}_sentiment.pkl")
            continue

        print(f"Encoding {len(raw)} headlines with FinBERT...")

        daily = build_daily_sentiment(raw)
        daily.to_pickle(f"data/{ticker}_sentiment.pkl")

        print(f"Saved: {len(daily)} daily sentiment records")
        print(daily[["date", "sentiment_score", "headline_count"]].head(10))