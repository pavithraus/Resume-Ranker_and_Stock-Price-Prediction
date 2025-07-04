""" 
install required packages

"""
!pip install yfinance keras scikit-learn matplotlib

"""**Import Modules and Set Seeds**"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
import datetime
import random

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

"""**Set reproducibility**"""

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

"""**Step 1 - Fetch Stock Data**"""

ticker = "TCS.NS"
end_date = datetime.datetime.today().strftime('%Y-%m-%d')
data = yf.download(ticker, start="2010-01-01", end=end_date)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
data.tail()

"""**Step 2 - Normalize Input Data**"""

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

"""**Step 3 - Prepare Sequences**"""

def prepare_sequences(dataset, label_index=3, window_size=120):
  x, y = [], []
  for i in range(window_size, len(dataset)):
    x.append(dataset[i - window_size:i])
    y.append(dataset[i, label_index])
  return np.array(x), np.array(y)

x_all, y_all = prepare_sequences(scaled, window_size=120)
train_len = int(len(x_all) * 0.8)
x_train, y_train = x_all[:train_len], y_all[:train_len]
x_test, y_test = x_all[train_len:], y_all[train_len:]

"""**Step 4 - Build the LSTM Model**"""

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(128))
model.add(Dropout(0.3))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

"""**Learning Rate Scheduler**"""

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

"""**Step 5 - Train the Model**



"""

history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=[reduce_lr],
    verbose=1
)

"""**Step 6 - Predictions and Evaluation**"""

predicted = model.predict(x_test)
predicted_actual = scaler.inverse_transform(
    np.concatenate([np.zeros((len(predicted), 3)),  # Open, High, Low dummy
                    predicted,
                    np.zeros((len(predicted), 1))], axis=1))[:, 3]

y_test_actual = scaler.inverse_transform(
    np.concatenate([np.zeros((len(y_test), 3)),
                    y_test.reshape(-1, 1),
                    np.zeros((len(y_test), 1))], axis=1))[:, 3]

"""**Step 7 - Evaluate the Model**"""

rmse = np.sqrt(mean_squared_error(y_test_actual, predicted_actual))
mae = mean_absolute_error(y_test_actual, predicted_actual)
r2 = r2_score(y_test_actual, predicted_actual)
print("\nModel Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.4f}")

"""**Step 8 - Train Predictions**"""

train_pred = model.predict(x_train)
train_pred_actual = scaler.inverse_transform(np.concatenate([np.zeros((len(train_pred), 3)), train_pred, np.zeros((len(train_pred), 1))], axis=1))[:, 3]
y_train_actual = scaler.inverse_transform(np.concatenate([np.zeros((len(y_train), 3)), y_train.reshape(-1, 1), np.zeros((len(y_train), 1))], axis=1))[:, 3]

train_r2 = r2_score(y_train_actual, train_actual)
print(f"Train R² Score: {train_r2:.4f}")

r2_test = r2_score(y_test_actual, predicted_actual)
print(f"Test R² Score: {r2_test:.4f}")

"""**Step 9 - Plot Actual vs Predicted Prices**"""

plt.figure(figsize=(14, 6))
plt.plot(y_test_actual, color='blue', label='Actual Price')
plt.plot(predicted_actual, color='red', label='Predicted Price')
plt.title(f"{ticker} Stock Price - Actual vs Predicted")
plt.xlabel("Time")
plt.ylabel("Price (INR)")
plt.legend()
plt.grid(True)
plt.show()

"""**Step 10 - Plot Training and Validation Loss**"""

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))

"""**Step 11 - Training Set: Actual vs Predicted**"""

plt.subplot(1, 2, 1)
plt.plot(y_train_actual, label='Actual (Train)', color='blue')
plt.plot(train_pred_actual, label='Predicted (Train)', color='orange')
plt.title("Training Set: Actual vs Predicted")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(y_test_actual, label='Actual (Test)', color='blue')
plt.plot(predicted_actual, label='Predicted (Test)', color='red')
plt.title("Test Set: Actual vs Predicted")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

"""**Step 12 - Test Set: Actual vs Predicted**

**Step 13 - Save model and scaler**
"""

model.save("lstm_model.h5")
np.save("scaler.npy", scaler.scale_)
np.save("min.npy", scaler.min_)
print("\nModel and scaler saved as lstm_model.h5, scaler.npy, min.npy")
