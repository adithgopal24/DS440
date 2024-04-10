import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import itertools

# Fetch and prepare data
def load_data(company, period):
    msft = yf.Ticker(company)
    hist = msft.history(period)
    hist.reset_index('Date', inplace=True)
    hist.insert(0, 'Name', company)
    hist.drop(columns = ['Dividends', 'Stock Splits'])
    hist = hist[['Name', 'Date', 'Open', 'Close', 'High', 'Low', 'Volume']]
    return hist

def fetch_stock_data(company, period):
    data = load_data(company, period)
    return data['Close']

def preprocess_data(stock_data, prediction_days):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data.reshape(-1, 1))
    
    x, y = [], []
    for i in range(prediction_days, len(scaled_data)):
        x.append(scaled_data[i-prediction_days:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(x), np.array(y)

def create_model(units, dropout, prediction_days):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(prediction_days, 1)),
        Dropout(dropout),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(units // 2),
        Dense(1)
    ])
    return model

# Train and evaluate model
def train_and_evaluate(model, epochs, batch_size, x_train, y_train, x_val, y_val):
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), verbose=0)
    predictions = model.predict(x_val)
    mae = mean_absolute_error(y_val, predictions)
    return mae

# Parameters for parameter testing
params = {
    'units': [32, 64, 128, 256],
    'dropout': [0.1, 0.2],
    'batch_size': [32, 64, 128],
    'prediction_days': [10, 20, 40]
}

# Load and preprocess data
COMPANY = 'TSLA'
PERIOD = '3mo'
data = fetch_stock_data(COMPANY, PERIOD).values

best_mae = float('inf')
best_params = {}

for units, dropout, batch_size, prediction_days in itertools.product(params['units'], params['dropout'], params['batch_size'], params['prediction_days']):
    x, y = preprocess_data(data, prediction_days)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    
    model = create_model(units, dropout, prediction_days)
    mae = train_and_evaluate(model, epochs=200, batch_size=batch_size, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
    
    if mae < best_mae:
        best_mae = mae
        best_params = {'units': units, 'dropout': dropout, 'batch_size': batch_size, 'prediction_days': prediction_days}
        print(f"New best parameters: {best_params} with MAE: {best_mae}")

print(f"Best parameters found: {best_params} with MAE: {best_mae}")

# Best parameters found: {'units': 128, 'dropout': 0.2, 'batch_size': 32, 'prediction_days': 40} with MAE: 0.030214642951191674