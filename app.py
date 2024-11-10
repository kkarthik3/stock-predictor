import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from model import create_dataset, build_model, train_model, delete_model_file  
import os
from keras.models import load_model

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Sidebar for uploading the CSV file
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
time_step = st.sidebar.number_input("Select time step (days):", min_value=1, max_value=1000, value=100)
months_to_predict = st.sidebar.number_input("Months to predict:", min_value=1, max_value=36, value=12)

if uploaded_file is not None:
    delete_model_file()  
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    data = df[['Date', 'Close']].set_index('Date')

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, index=data.index, columns=['Close'])
    
    # Train/Test split
    last_3_months = 3 * 100
    train_data = scaled_df[:-last_3_months]
    test_data = scaled_df[-last_3_months:]

    # Create datasets for LSTM
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Train model
    if st.sidebar.button("Train Model"):
        model = build_model(time_step)
        model = train_model(model, X_train, y_train, X_test, y_test, epochs=10)
        st.success("Model trained and saved as model.h5!")

    # Make predictions if model exists
    if os.path.exists("model.h5") and st.sidebar.button("Predict Future Prices"):
        model = load_model("model.h5")

        # Function to forecast future prices (adjusted for months to predict)
        def forecast_future(data, model, scaler, time_step, months_to_predict):
            last_days = data['Close'].to_list()[-time_step:]
            future_predictions = []
            for _ in range(months_to_predict * 20):  # Approx. trading days in a month
                reshaped_data = np.array(last_days[-time_step:]).reshape(1, time_step, 1)
                prediction = model.predict(reshaped_data)
                last_days.append(prediction[0][0])
                future_predictions.append(prediction[0][0])
            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
            return future_predictions

        future_predictions = forecast_future(scaled_df, model, scaler, time_step, months_to_predict)
        last_date = df['Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(future_predictions), freq='B')
        future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_predictions.flatten()})

        # Plot predictions
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df['Close'], mode='lines', name='Historical Close Price'))
        fig.add_trace(go.Scatter(x=future_df["Date"], y=future_df['Predicted_Close'], mode='lines', name='Predicted Close Price'))
        st.plotly_chart(fig)
else:
    st.warning("Please upload a CSV file to start.")
    st.image("https://paytmblogcdn.paytm.com/wp-content/uploads/2024/11/Blogs_Paytm_Bond-Market-vs.-Stock-Market_-Whats-the-Difference_-1.webp")
