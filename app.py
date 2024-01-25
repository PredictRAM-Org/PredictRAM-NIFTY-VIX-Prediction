import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

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

def plot_combined_candlestick_chart(df_nifty, df_vix, timeframe):
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

    # Add India VIX Candlestick with Blue for positive and Yellow for negative candles
    fig.add_trace(go.Candlestick(x=df_vix['timestamp'],
                                 open=df_vix_normalized['open'],
                                 high=df_vix_normalized['high'],
                                 low=df_vix_normalized['low'],
                                 close=df_vix_normalized['close'],
                                 name='India VIX',
                                 increasing=dict(line=dict(color='blue')),
                                 decreasing=dict(line=dict(color='yellow'))))

    fig.update_layout(title=f'Nifty 50 vs India VIX Candlestick Comparison ({timeframe})',
                      xaxis_title='Timestamp',
                      yaxis_title='Normalized Price',
                      xaxis_rangeslider_visible=False)

    st.plotly_chart(fig)

def main():
    st.title('Candlestick Chart Comparison')
    
    # Get user's choice for the timeframe
    timeframe = st.radio("Select Timeframe:", ["1 min", "30 min", "Day"])
    
    if timeframe == "1 min":
        api_url_nifty = "https://service.upstox.com/charts/v2/open/intraday/IN/NSE_INDEX|Nifty%2050/1minute/2024-01-25"
        api_url_vix = "https://service.upstox.com/charts/v2/open/intraday/IN/NSE_INDEX|India%20VIX/1minute/2024-01-24"
    elif timeframe == "30 min":
        api_url_nifty = "https://service.upstox.com/charts/v2/open/intraday/IN/NSE_INDEX|Nifty%2050/30minute/2024-01-25"
        api_url_vix = "https://service.upstox.com/charts/v2/open/historical/IN/NSE_INDEX|India%20VIX/30minute/2024-01-24"
    else:
        api_url_nifty = "https://service.upstox.com/charts/v2/open/historical/IN/NSE_INDEX|Nifty%2050/day/2024-01-25"
        api_url_vix = "https://service.upstox.com/charts/v2/open/historical/IN/NSE_INDEX|India%20VIX/day/2024-01-25"
    
    # Fetch and plot data
    df_nifty = fetch_data(api_url_nifty)
    df_vix = fetch_data(api_url_vix)
    plot_combined_candlestick_chart(df_nifty, df_vix, timeframe)

if __name__ == "__main__":
    main()
