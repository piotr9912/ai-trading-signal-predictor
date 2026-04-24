import yfinance as yf
import pandas as pd

print("Downloading data...")

df = yf.download("BTC-USD", period="1y")

print(df.head())
