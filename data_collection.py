import yfinance as yf
import pandas as pd
from transformers import pipeline
import talib
import numpy as np


def calculate_technical_indicators(data):
    data = data.astype(np.float64)
    
    data['rsi'] = talib.RSI(data['close'].values, timeperiod=14)
    data['macd'], data['macd_signal'], data['macd_hist'] = talib.MACD(data['close'].values)
    data['upper_band'], data['middle_band'], data['lower_band'] = talib.BBANDS(data['close'].values)
    data['volume_sma'] = talib.SMA(data['volume'].values, timeperiod=20)
    data['obv'] = talib.OBV(data['close'].values, data['volume'].values)
    data['adx'] = talib.ADX(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)
    data['cci'] = talib.CCI(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)
    
    return data.dropna()


def fetch_stock_data(ticker, period="2y", interval="1d"):
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=2)
    
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [col.lower() for col in data.columns]
    data = calculate_technical_indicators(data)
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
    
    return data.dropna()


def analyze_sentiment(news_headline):
    sentiment_pipeline = pipeline("text-classification", model="ProsusAI/finbert")
    result = sentiment_pipeline(news_headline)[0]
    
    # Get the label and confidence score
    sentiment = result['label']
    confidence = result['score']
    
    # Map sentiment to more readable format
    sentiment_map = {
        'positive': '🟢 Bullish',
        'negative': '🔴 Bearish',
        'neutral': '⚪ Neutral'
    }
    
    # Generate detailed analysis
    analysis = {
        'sentiment': sentiment_map.get(sentiment, sentiment),
        'confidence': f"{confidence * 100:.1f}%",
        'headline': news_headline,
        'explanation': generate_sentiment_explanation(sentiment, confidence, news_headline)
    }
    
    return analysis

def generate_sentiment_explanation(sentiment, confidence, headline):
    if sentiment == 'positive':
        if confidence > 0.8:
            return "The headline suggests strongly positive market sentiment, indicating potential upward momentum."
        else:
            return "The headline shows mildly positive sentiment, but with some uncertainty."
    elif sentiment == 'negative':
        if confidence > 0.8:
            return "The headline indicates significant bearish sentiment, suggesting potential downward pressure."
        else:
            return "The headline shows somewhat negative sentiment, but the signal isn't very strong."
    else:  # neutral
        return "The headline appears to have balanced or unclear implications for the market."
