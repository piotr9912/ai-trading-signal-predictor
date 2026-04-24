import pandas as pd

def add_indicators(df):
    df = df.copy()

    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

    df['ema9'] = df['Close'].ewm(span=9).mean()
    df['ema21'] = df['Close'].ewm(span=21).mean()

    df['return'] = df['Close'].pct_change()

    return df
