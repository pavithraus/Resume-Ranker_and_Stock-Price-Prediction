# 📈 LSTM-Based Stock Price Prediction.

An interactive stock prediction web app powered by an LSTM (Long Short-Term Memory) neural network. The app uses historical stock data to forecast future closing prices and is built using Streamlit for visualization.


---

### Built with the tools and technologies:
<p>
    <img src="https://img.shields.io/badge/-Streamlit-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/-Tensorflow-000000?style=for-the-badge&logo=markdown&logoColor=white" alt="Markdown">
    <img src="https://img.shields.io/badge/-ScikitLearn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
    <img src="https://img.shields.io/badge/-NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
    <img src="https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/-Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
    <img src="https://img.shields.io/badge/-Keras-00000?style=for-the-badge&logo=pandas&logoColor=white" alt="Keras">
</p>

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

### Stock Price Predictor
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
- Actual vs Predicted ![Actual vs Predicted](https://github.com/user-attachments/assets/18d823da-229b-4f87-8dcf-1de6ff93dfdb)

- Model Evaluation
- ![Model Evaluation](https://github.com/user-attachments/assets/87664c4d-573c-4586-ab8f-97fc205cc6af)

- Training vs Test ![Training vs Test](https://github.com/user-attachments/assets/dd7cded5-cde7-4ed8-835f-989a2244ed18)

- Training vs Validation Loss ![Training vs Validation Loss](https://github.com/user-attachments/assets/c6c442c0-ed8f-467e-98cc-278767f200b6)


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

---

## ✅ Conclusion

This project demonstrates how deep learning, specifically Long Short-Term Memory (LSTM) models, can be effectively used for time series forecasting in the stock market domain. By integrating a trained LSTM model with an interactive Streamlit application, users are empowered to explore stock data, visualize model performance, and generate future predictions in a flexible and accessible manner.

The application highlights key machine learning concepts such as data preprocessing, sequence modeling, model evaluation, and real-world deployment. While the predictions are educational and not financial advice, this project serves as a strong foundation for further enhancements such as integrating sentiment analysis, real-time updates, or ensemble modeling for improved accuracy.

Overall, the project provides a practical, hands-on approach to understanding the power and limitations of deep learning in financial forecasting.


