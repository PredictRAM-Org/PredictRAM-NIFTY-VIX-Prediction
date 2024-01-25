import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load the data
@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)

# LSTM model
def lstm_model(data):
    st.subheader("LSTM Model")

    # Assuming 'Close' is the column containing ^NSEI close prices
    series = data['Close'].values.reshape(-1, 1)

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series)

    # Create dataset
    def create_dataset(dataset, time_steps=1):
        X, y = [], []
        for i in range(len(dataset) - time_steps):
            a = dataset[i:(i + time_steps), 0]
            X.append(a)
            y.append(dataset[i + time_steps, 0])
        return np.array(X), np.array(y)

    time_steps = 5  # You can adjust the number of time steps
    X, y = create_dataset(scaled_data, time_steps)

    # Reshape input to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(1, time_steps)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=1)

    # Predict future values
    test_inputs = scaled_data[-time_steps:].reshape(1, -1)
    test_features = []
    for i in range(time_steps, 0, -1):
        test_features.append(test_inputs[:, i - time_steps:i])
    test_features = np.array(test_features)
    test_features = np.reshape(test_features, (test_features.shape[0], 1, test_features.shape[2]))

    predicted_scaled = model.predict(test_features)
    predicted_values = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))

    # Display forecast
    st.write("LSTM Forecast:")
    st.line_chart(pd.DataFrame({'Predicted': predicted_values.flatten()}))

# Linear Regression model
def linear_regression_model(data):
    st.subheader("Linear Regression Model")

    # Assuming 'Close' is the column containing ^NSEI close prices
    series = data['Close']

    # Create features (lagged values) for linear regression
    for i in range(1, 6):
        data[f'Lag_{i}'] = series.shift(i)

    # Drop NaN values
    data = data.dropna()

    # Split data into training and testing sets
    X = data.drop('Close', axis=1)
    y = data['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict future values
    test_features = X_test.tail(1)
    predicted_value = model.predict(test_features)

    # Display forecast
    st.write("Linear Regression Forecast:")
    st.line_chart(pd.DataFrame({'Predicted': [predicted_value[0]]}))

# ARIMA model
def arima_model(data):
    st.subheader("ARIMA Model")

    # Assuming 'Close' is the column containing ^NSEI close prices
    series = data['Close']

    # Fit ARIMA model
    order = (1, 1, 1)
    model = ARIMA(series, order=order)
    results = model.fit()

    # Forecast future values
    future_steps = 5  # You can adjust the number of future steps to predict
    forecast = results.get_forecast(steps=future_steps)

    # Display forecast
    st.write("ARIMA Forecast:")
    st.line_chart(pd.DataFrame({'Predicted': forecast.predicted_mean}))

# GARCH model
def garch_model(data):
    st.subheader("GARCH Model")

    # Assuming 'Close' is the column containing ^NSEI close prices
    series = data['Close']

    # Fit GARCH model
    model = arch_model(series, vol='Garch', p=1, q=1)
    results = model.fit()

    # Forecast future volatility
    forecast = results.forecast(horizon=5)  # You can adjust the horizon
    volatility_forecast = np.sqrt(forecast.variance.values[-1, :])

    # Display forecast
    st.write("GARCH Volatility Forecast:")
    st.line_chart(pd.DataFrame({'Volatility Forecast': volatility_forecast}))

# Streamlit UI
def main():
    st.title("Financial Market Prediction App")

    # Upload ^INDIAVIX.csv file
    uploaded_file = st.file_uploader("Choose ^INDIAVIX.csv file")
    if uploaded_file is not None:
        india_vix_data = load_data(uploaded_file)

        # Upload ^NSEI.NS file
        nsei_file = st.file_uploader("Choose ^NSEI.NS file")
        if nsei_file is not None:
            nsei_data = load_data(nsei_file)

            # Merge the two datasets on the 'Date' column
            merged_data = pd.merge(nsei_data, india_vix_data, on='Date', how='inner')

            # Show data summary
            st.write("Merged Data Summary:")
            st.write(merged_data.head())

            # Model selection
            model_option = st.selectbox("Select Predictive Model", ["LSTM", "Linear Regression", "ARIMA", "GARCH"])

            if model_option == "LSTM":
                lstm_model(merged_data)
            elif model_option == "Linear Regression":
                linear_regression_model(merged_data)
            elif model_option == "ARIMA":
                arima_model(merged_data)
            elif model_option == "GARCH":
                garch_model(merged_data)

if __name__ == "__main__":
    main()
