import yfinance as yf
import numpy as np
import pandas as pd
import math
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Parameters to iterate over
COMPANY = 'TSLA'
PERIODS = ['3mo', '6mo', '1y']
units_options = [32, 64, 128, 256]
dropout_options = [0.1, 0.2]
epochs = 200
batch_size_options = [32, 64, 128]
prediction_days = 30

# Function to load and preprocess data
def load_and_preprocess_data(company, period):
    data = yf.download(company, period=period)
    data.reset_index(inplace=True)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler, data

# Function to build LSTM model
def build_CNN_LSTM_model(input_shape, units, dropout_rate):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=units, return_sequences=True, input_shape=(input_shape[1], 1)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Placeholder for best parameters and their performance metrics
best_params = {}
best_performance = {'RMSE': float('inf')}

# Main loop to find best parameters
for period in PERIODS:
    x_train, y_train, scaler, data = load_and_preprocess_data(COMPANY, period)
    for units in units_options:
        for dropout_rate in dropout_options:
                for batch_size in batch_size_options:
                    model = build_CNN_LSTM_model(x_train.shape, units, dropout_rate)
                    # Split data into training and validation
                    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
                    model.fit(x_train_split, y_train_split, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(x_val_split, y_val_split))
                    # Validation data is being used as a proxy for test data
                    predictions = model.predict(x_val_split)
                    predictions = scaler.inverse_transform(predictions)
                    y_val_rescaled = scaler.inverse_transform(y_val_split.reshape(-1, 1))

                    # Calculate performance metrics
                    rmse = math.sqrt(mean_squared_error(y_val_rescaled, predictions))
                    mse = mean_squared_error(y_val_rescaled, predictions)
                    mae = mean_absolute_error(y_val_rescaled, predictions)
                    corr, _ = pearsonr(y_val_rescaled.reshape(-1), predictions.reshape(-1))

                    if rmse < best_performance['RMSE']:
                        best_performance = {'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'Correlation Coefficient': corr}
                        best_params = {'Company': COMPANY, 'Period': period, 'Units': units, 'Dropout Rate': dropout_rate, 'Epochs': epochs, 'Batch Size': batch_size}

# Due to the computational intensity of this process, especially with multiple epochs and larger units, this is a conceptual approach. In a real-world scenario, this process can take a significant amount of time and computational resources.

print(best_params)
print(best_performance)
