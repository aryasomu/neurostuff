import pandas as pd
import matplotlib.pyplot as plt
from data_collection import fetch_stock_data, analyze_sentiment
from strategy import backtest_strategy
from model_training import train_model, predict_next_day
from trade_analysis import generate_trade_report
import streamlit as st
import os
import webbrowser
import subprocess
import time
import sys

def main():
    print("🔹 Welcome to Synapse MVP 🔹\n")
    ticker = input("Enter stock ticker (e.g., AAPL, TSLA, MSFT): ").upper()
    print(f"\n📈 Fetching stock data for {ticker}...")
    data = fetch_stock_data(ticker)
    print(data.head())
    print("\n🤖 Training prediction model...")
    model, scaler, feature_columns = train_model(ticker)
    
    print("\n🔮 Making prediction for next day...")
    try:
        prediction = predict_next_day(model, scaler, ticker, feature_columns)
        if prediction is None:
            print("⚠️ Could not make prediction for next day")
    except Exception as e:
        print(f"⚠️ Error making prediction: {str(e)}")
    
    news_headline = input("\nEnter a stock-related news headline for sentiment analysis: ")
    sentiment_result = analyze_sentiment(news_headline)
    
    print("\n🧠 Sentiment Analysis Result:")
    print("=" * 50)
    print(f"Headline: {sentiment_result['headline']}")
    print(f"Sentiment: {sentiment_result['sentiment']}")
    print(f"Confidence: {sentiment_result['confidence']}")
    print(f"Analysis: {sentiment_result['explanation']}")
    print("=" * 50)
    
    print("\n📊 Running backtest and generating trade report...\n")
    trades, backtest_data = backtest_strategy(data)
    
    print("\n📝 Generating detailed trade analysis report...")
    
    # Save the data for Streamlit
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv('temp_trades.csv', index=False)
    data.to_csv('temp_data.csv')
    
    print("\n🚀 Launching trade analysis dashboard...")
    print("Please wait while the dashboard loads in your browser...")
    
    # Simple Streamlit launch
    try:
        # Open browser after a short delay
        def open_browser():
            time.sleep(2)
            webbrowser.open('http://localhost:8501')
        
        import threading
        threading.Thread(target=open_browser).start()
        
        # Run Streamlit directly
        os.system('streamlit run dashboard.py')
        
    except KeyboardInterrupt:
        print("\n\n👋 Closing dashboard...")
        # Clean up temporary files
        if os.path.exists('temp_trades.csv'):
            os.remove('temp_trades.csv')
        if os.path.exists('temp_data.csv'):
            os.remove('temp_data.csv')
        print("Goodbye!")
    except Exception as e:
        print(f"\n⚠️ Error launching dashboard: {str(e)}")
        print("You can try running the dashboard manually with: streamlit run dashboard.py")

if __name__ == "__main__":
    main()
