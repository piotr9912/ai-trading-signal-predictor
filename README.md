# ai-trading-signal-predictor
 Machine learning model for predicting market direction using technical indicators Visibility: Public

## Model Prediction

![Prediction Plot](charts/prediction_plot.png)

## Results

- Model Accuracy: ~0.55 - 0.60
- Task: Predict next-day market direction (BTC)
- Features: EMA(9), EMA(21), Returns

The model shows predictive capability above random baseline (~50%).

## How it works

1. Download historical market data using Yahoo Finance
2. Generate technical indicators (EMA, returns)
3. Create classification target (price up/down)
4. Train a Random Forest model
5. Evaluate predictions and visualize results

## Project Goal

The goal of this project is to demonstrate a full AI pipeline:
data collection → feature engineering → model training → prediction → visualization.

It is designed as a foundation for quantitative trading and AI-driven financial models.
