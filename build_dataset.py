# build_dataset.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA

# ── Constants ──────────────────────────────────────────────────────────────────
WINDOW   = 20
TICKERS  = ["AAPL", "TSLA"]
LAMBDA_P = 0.05   # price decay
LAMBDA_S = 0.30   # sentiment decay
SENT_DIM = 33     # 1 score + 32 PCA components

os.makedirs("data", exist_ok=True)

# ── Step 1: Fit PCA once ───────────────────────────────────────────────────────
if not os.path.exists("data/pca.pkl"):
    print("Fitting PCA on all embeddings...")
    all_embeddings = []
    for ticker in TICKERS:
        sent = pd.read_pickle(f"data/{ticker}_sentiment.pkl")
        embs = np.stack(sent["embedding"].values)
        all_embeddings.append(embs)
    all_embeddings = np.vstack(all_embeddings)
    pca = PCA(n_components=32, random_state=42)
    pca.fit(all_embeddings)
    joblib.dump(pca, "data/pca.pkl")
    print(f"PCA explains {pca.explained_variance_ratio_.sum()*100:.1f}% variance")
else:
    print("Loading existing PCA...")

pca = joblib.load("data/pca.pkl")

# ── Step 2: Load and prepare sentiment data ────────────────────────────────────
print("\nLoading sentiment data...")
sent_data = {}
for ticker in TICKERS:
    df = pd.read_pickle(f"data/{ticker}_sentiment.pkl")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df["pca_emb"] = df["embedding"].apply(
        lambda e: pca.transform(e.reshape(1, -1))[0]
        if isinstance(e, np.ndarray) else np.zeros(32)
    )
    sent_data[ticker] = df
    print(f"  {ticker}: {len(df)} sentiment days")

# ── Step 3: Load price data ────────────────────────────────────────────────────
print("\nLoading price data...")
price_data = {}
for ticker in TICKERS:
    df = pd.read_csv(f"data/{ticker}_prices.csv",
                     index_col=0, parse_dates=True)
    price_data[ticker] = df
    print(f"  {ticker}: {len(df)} trading days")

PRICE_COLS = ["open","high","low","close","volume",
              "rsi","macd","macd_sig","ema_20","ema_50","bb_width"]

# ── Step 4: Build sequences ────────────────────────────────────────────────────
def get_sent_vec(ticker, date):
    """Returns 33-d sentiment vector [score, pca_0..pca_31] for a given date."""
    if date in sent_data[ticker].index:
        row = sent_data[ticker].loc[date]
        return np.concatenate([[row["sentiment_score"]], row["pca_emb"]])
    return np.zeros(SENT_DIM)

def build_sequences(ticker):
    """
    Returns:
        X_price : [N, WINDOW, 11]          — price + technical features
        X_sent  : [N, WINDOW, num_stocks, SENT_DIM]  — sentiment for ALL stocks
        y       : [N]                       — next-day direction label
    """
    prices = price_data[ticker]
    dates  = prices.index.tolist()
    n_stocks = len(TICKERS)

    X_price, X_sent, y_list = [], [], []

    for i in range(WINDOW, len(dates) - 1):
        window_dates = dates[i - WINDOW : i]

        price_seq = []   # [WINDOW, 11]
        sent_seq  = []   # [WINDOW, n_stocks, SENT_DIM]

        for j, d in enumerate(window_dates):
            days_ago = WINDOW - j

            # ── Price features with temporal decay ──
            p_row = prices.loc[d, PRICE_COLS].values.astype(float)
            price_seq.append(p_row * np.exp(-LAMBDA_P * days_ago))

            # ── Sentiment for every stock with temporal decay ──
            # Shape: [n_stocks, SENT_DIM]
            t_sent = []
            for t in TICKERS:
                s_vec = get_sent_vec(t, d)
                t_sent.append(s_vec * np.exp(-LAMBDA_S * days_ago))
            sent_seq.append(t_sent)

        X_price.append(np.array(price_seq))          # [WINDOW, 11]
        X_sent.append(np.array(sent_seq))             # [WINDOW, n_stocks, SENT_DIM]
        y_list.append(int(prices.iloc[i]["label"]))

    X_price = np.array(X_price)   # [N, WINDOW, 11]
    X_sent  = np.array(X_sent)    # [N, WINDOW, n_stocks, SENT_DIM]
    y       = np.array(y_list)    # [N]
    return X_price, X_sent, y

# ── Step 5: Save all tensors ───────────────────────────────────────────────────
print("\nBuilding sequences...")
for ticker in TICKERS:
    X_price, X_sent, y = build_sequences(ticker)

    np.save(f"data/{ticker}_Xprice.npy", X_price)
    np.save(f"data/{ticker}_Xsent.npy",  X_sent)
    np.save(f"data/{ticker}_y.npy",      y)

    print(f"  {ticker}:")
    print(f"    X_price : {X_price.shape}   (N, window, price_features)")
    print(f"    X_sent  : {X_sent.shape}  (N, window, stocks, sent_dim)")
    print(f"    y       : {y.shape},  class balance={y.mean():.2f}")
