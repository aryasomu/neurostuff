import streamlit as st
import pandas as pd
from trade_analysis import generate_trade_report
import os
import time

def main():
    st.set_page_config(
        page_title="Synapse Trade Analysis",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Synapse Trade Analysis Dashboard")
    
    # Add a loading spinner while checking for data
    with st.spinner("Checking for trade data..."):
        time.sleep(1)  # Give time for files to be written
        
        if not os.path.exists('temp_trades.csv') or not os.path.exists('temp_data.csv'):
            st.error("❌ No trade data found. Please run the main application first to generate trade data.")
            st.info("💡 Run `python main.py` in your terminal to generate the trade data.")
            return
            
        try:
            trades = pd.read_csv('temp_trades.csv')
            if trades.empty:
                st.warning("⚠️ No trades were executed during this period.")
                return
                
            data = pd.read_csv('temp_data.csv')
            
            # Properly set the datetime index for data
            if 'Date' in data.columns:
                data.set_index('Date', inplace=True)
            data.index = pd.to_datetime(data.index)
            
            # Convert trades dates to datetime
            if 'date' in trades.columns:
                trades['date'] = pd.to_datetime(trades['date'])
            
            # Generate the trade report
            generate_trade_report(trades.to_dict('records'), data)
            
        except Exception as e:
            st.error(f"❌ An error occurred while loading the data: {str(e)}")
            st.write("Please try the following:")
            st.write("1. Run the main application again")
            st.write("2. Make sure you have entered a valid stock ticker")
            st.write("3. Check that the data files were generated properly")

if __name__ == "__main__":
    main()
