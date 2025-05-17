# Import Required Libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import Technical Indicators Explicitly
from ta.momentum import RSIIndicator
from ta.trend import MACD


def get_stock_data(ticker, start="2010-01-01", end="2024-03-02"):
    """
    Fetch historical stock market data from Yahoo Finance.
    """
    df = yf.download(ticker, start=start, end=end, interval="1d")
    return df


def add_technical_indicators(df_):
    """
    Add Moving Averages, RSI, and MACD indicators to the dataset.
    """
    df_ = df_.copy()  # Prevent modifying the original dataframe

    # Compute Moving Averages
    df_["SMA_50"] = df_["Close"].rolling(window=50).mean()
    df_["SMA_200"] = df_["Close"].rolling(window=200).mean()

    # ✅ FIXED RSI Calculation
    df_["RSI"] = RSIIndicator(df_["Close"], window=14).rsi().squeeze()
    df_["RSI"] = df_["RSI"].astype(float)  # Ensure it is a float type

    # Compute MACD
    macd = MACD(df_["Close"])
    df_["MACD"] = macd.macd()
    df_["MACD_Signal"] = macd.macd_signal()

    # Drop NaN values after adding indicators
    df_.dropna(inplace=True)

    return df_


# Load Data
df_main = get_stock_data("AAPL")  # Fetch Apple stock data
df_main = add_technical_indicators(df_main)  # Apply technical indicators

# Create Target Variable (Predicting Returns Instead of Raw Price)
df_main["Return_Tomorrow"] = df_main["Close"].pct_change().shift(-1)  # Predict percentage change
df_main.dropna(inplace=True)

# Select Features & Target
features = ["Open", "High", "Low", "Close", "Volume", "SMA_50", "SMA_200", "RSI", "MACD", "MACD_Signal"]
X = df_main[features]
y = df_main["Return_Tomorrow"]

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale Data (Important for XGBoost Performance)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost Model (Better for Stock Prediction)
model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
model.fit(X_train_scaled, y_train)

# Make Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"📉 Mean Absolute Error (MAE): {mae:.4f}")
print(f"📉 Root Mean Squared Error (RMSE): {rmse:.4f}")

# Feature Importance Check
importance_values = model.feature_importances_
feature_names = X.columns
sorted_indices = np.argsort(importance_values)[::-1]

print("\n📊 Feature Importance Ranking:")
for i in sorted_indices:
    print(f"{feature_names[i]}: {importance_values[i]:.4f}")

# Plot Predictions
plt.figure(figsize=(12, 6))

x_values = df_main.index[-len(y_test):]  # Ensure x-axis length matches y_test
y_actual = y_test.values  # Convert y_test to NumPy array
y_predicted = y_pred[:len(y_test)]  # Ensure predictions match y_test

plt.plot(x_values, y_actual, label="Actual Returns", color="blue", linewidth=2)
plt.plot(x_values, y_predicted, label="Predicted Returns", color="red", linewidth=1)

plt.xlabel("Date")
plt.ylabel("Stock Return (%)")
plt.title("Stock Return Prediction (XGBoost)")
plt.legend()
plt.show()
