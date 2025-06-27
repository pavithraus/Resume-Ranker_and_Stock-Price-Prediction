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

###stock_forecast_app
1. app
   - lstm_app.py # Streamlit dashboard application
   - stock_predictor.py # LSTM training script

2. models
   - lstm_model.keras # Trained LSTM model
   - scaler.npy # Scaler used during training
   - min.npy # Optional normalization min values

3. assets
   - Actual vs Predicted.png # Comparison of actual vs predicted prices
   - Model Evaluation.PNG # Summary of model performance metrics like R², RMSE, and MAE.
   - Training vs Test.png # Line plot comparing predicted vs actual values on training and test datasets
   - Training vs Validation Loss.png # training and validation loss evolve to detect overfitting or underfitting
     
4. recordings/
   -  streamlit_demo.mp4 # Screen recording of the Streamlit app

5. requirements.txt # Required Python packages

6.  README.md # Project documentation
---
## 🧾 Requirements

Main packages (from requirements.txt):

- streamlit
- numpy
- pandas
- keras
- tensorflow
- matplotlib
- scikit-learn
- yfinance

Make sure to use compatible versions (e.g., TensorFlow 2.11, Keras 2.11) and NumPy < 2.0 for best stability.

---

## 📸 Screenshots & Visuals
- Actual vs Predicted
- Model Evaluation
- Training vs Test
- Training vs Validation Loss

---
## 🎥 Streamlit Demo
- Local Host
---
## ✨ Features

- 📦 Pre-trained LSTM model on stock data

- 📂 Option to upload your own CSV

- 🔁 Forecasting Modes:
  - Next N business days
  - Specific future date
  - Custom date range

- 📤 Download predictions as CSV

- 📈 Visual comparison of predictions vs real data

- 📊 Metrics: R² Score, RMSE, MAE

