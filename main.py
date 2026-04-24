import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from features.indicators import add_indicators
from model.train import train_model
from predict.inference import predict

print("Downloading data...")

df = yf.download("BTC-USD", period="1y")

df = add_indicators(df)

df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

df = df.dropna()

features = ['ema9', 'ema21', 'return']

X = df[features]
y = df['target']

split = int(len(df) * 0.8)

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

model = train_model(X_train, y_train)

preds = predict(model, X_test)

accuracy = (preds == y_test).mean()

print("Model accuracy:", accuracy)

# 🔥 WYKRES
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual")
plt.plot(preds, label="Predicted", alpha=0.7)

plt.title("Prediction vs Actual")
plt.xlabel("Time")
plt.ylabel("Direction (0/1)")
plt.legend()

# 🔥 ZAPIS DO PLIKU
plt.savefig("charts/prediction_plot.png")

plt.show()
