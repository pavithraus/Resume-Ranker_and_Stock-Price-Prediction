# 📈 LSTM-Based Stock Price Prediction

An interactive stock prediction web app powered by an LSTM (Long Short-Term Memory) neural network. The app uses historical stock data to forecast future closing prices and is built using Streamlit for visualization.

---

## 🧠 Project Overview

-  Predicts stock closing prices using an LSTM model.
-  Trained on Yahoo Finance data for selected companies (e.g., TCS.NS).
-  Provides an interactive Streamlit dashboard with options:
  - Forecast next N days
  - Forecast a specific future date
  - Forecast a custom date range
-  Visualizes actual vs predicted prices
-  Allows data upload and CSV export of results

---

## 📁 Folder Structure

stock_forecast_app/
├── app/
│ ├── lstm_app.py # Streamlit dashboard application
│ ├── stock_predictor.py # LSTM training script
│
├── models/
│ ├── lstm_model.keras # Trained LSTM model
│ ├── scaler.npy # Scaler used during training
│ ├── min.npy # Optional normalization min values
│
├── assets/
│ └── training_visuals/
│ ├── loss_curve.png # Model training loss curve
│ ├── actual_vs_pred.png # Comparison of actual vs predicted prices
│
├── recordings/
│ └── streamlit_demo.mp4 # Screen recording of the Streamlit app
│
├── requirements.txt # Required Python packages
├── README.md # Project documentation

