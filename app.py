import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

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

# Streamlit UI
def main():
    st.title("Financial Market Prediction App")

    # Upload ^INDIAVIX.csv file
    vix_file = st.file_uploader("Choose ^INDIAVIX.csv file")
    if vix_file is not None:
        vix_data = load_data(vix_file)

        # Upload ^NSEI.csv file
        nsei_file = st.file_uploader("Choose ^NSEI.csv file")
        if nsei_file is not None:
            nsei_data = load_data(nsei_file)

            # Merge the two datasets on the 'Date' column
            merged_data = pd.merge(nsei_data, vix_data, on='Date', how='inner')

            # Show data summary
            st.write("Merged Data Summary:")
            st.write(merged_data.head())

            # Model selection
            model_option = st.selectbox("Select Predictive Model", ["LSTM"])

            if model_option == "LSTM":
                lstm_model(merged_data)

if __name__ == "__main__":
    main()
