import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands

TICKERS = ["AAPL", "TSLA"]
START   = "2019-01-01"
END     = date.today().strftime("%Y-%m-%d")

def get_price_features(ticker):
    df = yf.download(ticker, start=START, end=END, auto_adjust=True)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]

    # Technical indicators
    df["rsi"]     = RSIIndicator(df["close"], window=14).rsi()
    df["macd"]    = MACD(df["close"]).macd()
    df["macd_sig"]= MACD(df["close"]).macd_signal()
    df["ema_20"]  = EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema_50"]  = EMAIndicator(df["close"], window=50).ema_indicator()
    bb = BollingerBands(df["close"], window=20)
    df["bb_upper"]= bb.bollinger_hband()
    df["bb_lower"]= bb.bollinger_lband()
    df["bb_width"]= (df["bb_upper"] - df["bb_lower"]) / df["close"]

    # Label: next-day direction (1=up, 0=down/flat)
    df["return"]  = df["close"].pct_change().shift(-1)
    df["label"]   = (df["return"] > 0).astype(int)

    df.dropna(inplace=True)
    df["ticker"] = ticker
    return df

price_data = {t: get_price_features(t) for t in TICKERS}
for t, df in price_data.items():
    df.to_csv(f"data/{t}_prices.csv")
    print(f"{t}: {len(df)} trading days")