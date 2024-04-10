import yfinance as yf
import numpy as np
import pandas as pd
import math
from scipy.stats import pearsonr

import os
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def load_data(company, period):
    msft = yf.Ticker(company)
    hist = msft.history(period)
    hist.reset_index('Date', inplace=True)
    hist.insert(0, 'Name', company)
    hist.drop(columns = ['Dividends', 'Stock Splits'])
    hist = hist[['Name', 'Date', 'Open', 'Close', 'High', 'Low', 'Volume']]
    return hist

COMPANY = 'TSLA'
PERIOD = "3mo"
################ need to change COMPANY & PERIOD ################
# 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, and ytd

# Check if COMPANY name is valid
tickers = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

if (tickers.Symbol == COMPANY).any().any():
    #print('valid')
    pass
else:
    print('not valid')
    exit()

data = load_data(COMPANY, PERIOD)

# Normalize data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

# Set the number of days used for prediction
prediction_days = 40
### 10000 -> error

# Initialize empty lists for training data input and output
x_train = []
y_train = []

# Iterate through the scaled data, starting from the prediction_days index
for x in range(prediction_days, len(scaled_data)):
    # Append the previous 'prediction_days' values to x_train
    x_train.append(scaled_data[x - prediction_days:x, 0])
    # Append the current value to y_train
    y_train.append(scaled_data[x, 0])

# Convert the x_train and y_train lists to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape x_train to a 3D array with the appropriate dimensions for the LSTM model
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

def LSTM_model():
    # Initialize a sequential model
    model = Sequential()

    # Add the first LSTM layer with 50 units, input shape, and return sequences
    model.add(LSTM(units=128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    # Add dropout to prevent overfitting
    model.add(Dropout(0.2))

    # Add a second LSTM layer with 50 units and return sequences
    model.add(LSTM(units=128, return_sequences=True))
    # Add dropout to prevent overfitting
    model.add(Dropout(0.2))

    # Add a third LSTM layer with 50 units
    model.add(LSTM(units=128))
    # Add dropout to prevent overfitting
    model.add(Dropout(0.2))

    # Add a dense output layer with one unit
    model.add(Dense(units=1))

    return model

model = LSTM_model()
model.summary()
model.compile(
    optimizer='adam', 
    loss='mean_squared_error'
)

# Define callbacks

# Save weights only for best model
checkpointer = ModelCheckpoint(
    filepath = 'weights_best.hdf5', 
    verbose = 2, 
    save_best_only = True
)
'''
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)
history = model.fit(
    x_train, 
    y_train, 
    epochs=200,  # Or however many epochs you desire
    batch_size=32,
    validation_data=(x_val, y_val),  # Include validation data here
    callbacks=[checkpointer]
)
# Plot training & validation loss values
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()
'''
model.fit(
    x_train, 
    y_train, 
    epochs=200,
    ## too much epochs slows down the runtime 
    batch_size = 32,
    callbacks = [checkpointer]
)

# Load test data for the specified company and date range
test_data = load_data(COMPANY, PERIOD)

# Extract the actual closing prices from the test data
actual_prices = test_data['Close'].values

# Concatenate the training and test data along the 'Close' column
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

# Extract the relevant portion of the dataset for model inputs
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values

# Reshape the model inputs to a 2D array with a single column
model_inputs = model_inputs.reshape(-1, 1)

# Apply the same scaling used for training data to the model inputs
model_inputs = scaler.transform(model_inputs)

# Initialize an empty list for test data input
x_test = []

# Iterate through the model inputs, starting from the prediction_days index
for x in range(prediction_days, len(model_inputs)):
    # Append the previous 'prediction_days' values to x_test
    x_test.append(model_inputs[x-prediction_days:x, 0])

# Convert the x_test list to a numpy array
x_test = np.array(x_test)

# Reshape x_test to a 3D array with the appropriate dimensions for the LSTM model
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Generate price predictions using the LSTM model
predicted_prices = model.predict(x_test)

# Invert the scaling applied to the predicted prices to obtain actual values
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the actual prices using a black line
plt.plot(actual_prices, color='black', label=f"Actual {COMPANY} price")

# Plot the predicted prices using a green line
plt.plot(predicted_prices, color='green', label=f"Predicted {COMPANY} price")

# Set the title of the plot using the company name
plt.title(f"{COMPANY} share price")

# Set the x-axis label as 'time'
plt.xlabel("time")

# Set the y-axis label using the company name
plt.ylabel(f"{COMPANY} share price")

# Display a legend to differentiate the actual and predicted prices
plt.legend()

# Show the plot on the screen
plt.show()

# Extract the last 'prediction_days' values from the model inputs
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]

# Convert the real_data list to a numpy array
real_data = np.array(real_data)

# Reshape real_data to a 3D array with the appropriate dimensions for the LSTM model
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

# Generate a prediction using the LSTM model with the real_data input
prediction = model.predict(real_data)

# Invert the scaling applied to the prediction to obtain the actual value
prediction = scaler.inverse_transform(prediction)

# Print the most recent stock price
stock_data = yf.download(COMPANY, period='1d')
most_recent_close = stock_data['Close'][-1]
print(f"The most recent closing price for {COMPANY} is ${most_recent_close}")

# Print the prediction result to the console
print(f"Prediction: {prediction[0][0]}")

# Whether increse or decrease
if most_recent_close > prediction[0][0]:
    print(f"It will decrease by ${most_recent_close - prediction[0][0]}")
else:
    print(f"It will increase by ${prediction[0][0] - most_recent_close}")

# Calculate MSE (better closer to zero)
mse = mean_squared_error(actual_prices, predicted_prices)
print(f"Mean Squared Error: {mse}")

# Calculate RMSE (better closer to zero)
rmse = math.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Calculate MAE (better closer to zero)
mae = mean_absolute_error(actual_prices, predicted_prices)
print(f"Mean Absolute Error: {mae}")

# Calculate Correlation Coefficient (better closer to 1)
corr, _ = pearsonr(actual_prices.reshape(-1), predicted_prices.reshape(-1))
print(f"Correlation Coefficient: {corr}")