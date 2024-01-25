import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime

# Load the data
@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)

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
    st.write(forecast.predicted_mean)

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
    st.write(volatility_forecast)

# Machine Learning model
def ml_model(data):
    st.subheader("Machine Learning Model")

    # Assuming 'Close' is the column containing ^NSEI close prices
    target_column = 'Close'

    # Choose features for the model
    features = data.drop(target_column, axis=1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, data[target_column], test_size=0.2, random_state=42)

    # Build and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on future data (you need to adjust this based on your actual data)
    future_data = X_test.head(1)  # Using the first row as an example
    prediction = model.predict(future_data)

    # Display prediction
    st.write("Machine Learning Prediction:")
    st.write(prediction)

# Streamlit UI
def main():
    st.title("Financial Market Prediction App")

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose ^INDIAVIX.csv file")
    if uploaded_file is not None:
        india_vix_data = load_data(uploaded_file)

        # Show data summary
        st.write("INDIAVIX Data Summary:")
        st.write(india_vix_data.head())

        # Model selection
        model_option = st.selectbox("Select Predictive Model", ["ARIMA", "GARCH", "Machine Learning"])

        if model_option == "ARIMA":
            arima_model(india_vix_data)
        elif model_option == "GARCH":
            garch_model(india_vix_data)
        elif model_option == "Machine Learning":
            ml_model(india_vix_data)

if __name__ == "__main__":
    main()
