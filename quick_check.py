# quick_check.py
import pandas as pd
import numpy as np

for t in ["AAPL", "TSLA"]:
    df = pd.read_pickle(f"data/{t}_sentiment.pkl")
    print(f"\n{t}: {len(df)} days with sentiment")
    print(f"  Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"  Avg headlines/day: {df['headline_count'].mean():.1f}")
    print(f"  Sentiment score range: {df['sentiment_score'].min():.3f} "
          f"to {df['sentiment_score'].max():.3f}")
    print(f"  Embedding shape check: {df['embedding'].iloc[0].shape}")
    # Should print (768,)