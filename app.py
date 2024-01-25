import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import time

def fetch_data(api_url):
    response = requests.get(api_url)
    data = response.json()

    # Assuming the data structure contains a key 'data' which contains 'candles'
    df = pd.DataFrame(data['data']['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'unknown_column'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]  # Extract relevant columns
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Convert timestamp to datetime
    return df

def normalize_data(df):
    scaler = MinMaxScaler()
    df[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])
    return df

def plot_combined_candlestick_chart(df_nifty, df_vix, title):
    fig = go.Figure()

    # Normalize the data for proper comparison
    df_nifty_normalized = normalize_data(df_nifty)
    df_vix_normalized = normalize_data(df_vix)

    # Add Nifty 50 Candlestick
    fig.add_trace(go.Candlestick(x=df_nifty['timestamp'],
                                 open=df_nifty_normalized['open'],
                                 high=df_nifty_normalized['high'],
                                 low=df_nifty_normalized['low'],
                                 close=df_nifty_normalized['close'],
                                 name='Nifty 50'))

    # Add India VIX Candlestick
    fig.add_trace(go.Candlestick(x=df_vix['timestamp'],
                                 open=df_vix_normalized['open'],
                                 high=df_vix_normalized['high'],
                                 low=df_vix_normalized['low'],
                                 close=df_vix_normalized['close'],
                                 name='India VIX',
                                 increasing=dict(line=dict(color='blue')),
                                 decreasing=dict(line=dict(color='yellow'))))

    fig.update_layout(title=title,
                      xaxis_title='Timestamp',
                      yaxis_title='Normalized Price',
                      xaxis_rangeslider_visible=False)

    st.plotly_chart(fig)

def main():
    st.title('Live 1-Minute Candlestick Chart Comparison')

    api_url_nifty = "https://service.upstox.com/charts/v2/open/intraday/IN/NSE_INDEX|Nifty%2050/1minute/2024-01-25"
    api_url_vix = "https://service.upstox.com/charts/v2/open/intraday/IN/NSE_INDEX|India%20VIX/1minute/2024-01-25"
    
    refresh_button = st.button("Refresh Chart")

    while True:
        if refresh_button:
            st.success("Chart is refreshed!")

        # Fetch and plot data
        df_nifty = fetch_data(api_url_nifty)
        df_vix = fetch_data(api_url_vix)
        plot_combined_candlestick_chart(df_nifty, df_vix, 'Live 1-Minute Candlestick Chart Comparison')

        # Pause for 1 minute before refreshing data
        time.sleep(60)

        # Clear the refresh button state
        refresh_button = False

if __name__ == "__main__":
    main()
