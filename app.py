import pandas as pd
import yfinance as yf
import ta
from ta import add_all_ta_features
from ta.utils import dropna
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.tools as tls
import streamlit as st
from datetime import date
import datetime
from datetime import datetime, date
from streamlit_option_menu import option_menu
import io
import numpy as np
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


yf.pdr_override()
#st.set_option('deprecation.showfileUploaderEncoding', False)

st.markdown("<h1 style='text-align: center; color: green;'>StockGrader.io</h1>", unsafe_allow_html=True)

time = pd.to_datetime('now')
today_val = date.today()
# Get S&P 500 constituents
sp500 = yf.Tickers("^GSPC")
# Extract tickers and put them in a list
tickers = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers_names = tickers.Symbol.to_list()

def basic_user_input_features():
    #stock_choice = st.radio("Pick a Stock to see its information: ", [tickers_names])
    stock_choice = st.selectbox('Select Stock to See its info', tickers_names)
    #stock_choice = st.radio("Pick a Stock to see its information: ", [names])
    ticker = st.sidebar.text_input("Selected Stock Ticker", stock_choice)
    start_date = st.sidebar.text_input("Start Date", '2019-01-01')
    end_date = st.sidebar.text_input("End Date", f'{today_val}')
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
    buying_price = st.sidebar.number_input("Buying Price", value=0.2000, step=0.0001)
    balance = st.sidebar.number_input("Quantity", value=0.0, step=0.0001)
    #file_buffer = st.sidebar.file_uploader("Choose a .csv or .xlxs file\n 2 columns are expected 'rate' and 'price'", type=['xlsx','csv'])
    if start_date_obj > today_val:
        return st.warning('The Start Date is a date in the future, and therefore is not valid. Please adjust it.', icon="⚠️")
    elif end_date_obj > today_val:
        return st.warning('The End Date is a date in the future, and therefore is not valid. Please adjust it.', icon="⚠️")
    elif start_date_obj > end_date_obj:
        return st.warning('The End Date is a date coming before the Start Date. Please adjust the date range.', icon="⚠️")
    return ticker, start_date, end_date, buying_price, balance

def advanced_user_input_features():
    pass
def load_data(company, period = '3mo'):
    msft = yf.Ticker(company)
    hist = msft.history(period)
    hist.reset_index('Date', inplace=True)
    hist.insert(0, 'Name', company)
    hist.drop(columns=['Dividends', 'Stock Splits'])
    hist = hist[['Name', 'Date', 'Open', 'Close', 'High', 'Low', 'Volume']]
    return hist
def provide_LSTM_model(company, period = '3mo'):
    data = load_data(company, period)

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Set the number of days used for prediction
    prediction_days = 30
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

    # Initialize a sequential model
    model = Sequential()

    # Add the first LSTM layer with 50 units, input shape, and return sequences
    model.add(LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    # Add dropout to prevent overfitting
    model.add(Dropout(0.1))

    # Add a second LSTM layer with 50 units and return sequences
    model.add(LSTM(units=64, return_sequences=True))
    # Add dropout to prevent overfitting
    model.add(Dropout(0.1))

    # Add a third LSTM layer with 50 units
    model.add(LSTM(units=64))
    # Add dropout to prevent overfitting
    model.add(Dropout(0.1))

    # Add a dense output layer with one unit
    model.add(Dense(units=1))

    lstm_model = model
    lstm_model.summary()
    lstm_model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )
    checkpointer = ModelCheckpoint(
        filepath='weights_best.hdf5',
        verbose=2,
        save_best_only=True
    )
    lstm_model.fit(
        x_train,
        y_train,
        epochs=200,
        ## too much epochs slows down the runtime
        batch_size=32,
        callbacks=[checkpointer]
    )

    # Load test data for the specified company and date range
    test_data = load_data(company, period)

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
        x_test.append(model_inputs[x - prediction_days:x, 0])

    # Convert the x_test list to a numpy array
    x_test = np.array(x_test)

    # Reshape x_test to a 3D array with the appropriate dimensions for the LSTM model
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Generate price predictions using the LSTM model
    predicted_prices = lstm_model.predict(x_test)

    # Invert the scaling applied to the predicted prices to obtain actual values
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Plot the actual prices using a black line
    plt.plot(actual_prices, color='black', label=f"Actual {company} price")

    # Plot the predicted prices using a green line
    plt.plot(predicted_prices, color='green', label=f"Predicted {company} price")

    # Set the title of the plot using the company name
    plt.title(f"{company} share price")

    # Set the x-axis label as 'time'
    plt.xlabel("time")

    # Set the y-axis label using the company name
    plt.ylabel(f"{company} share price")

    # Display a legend to differentiate the actual and predicted prices
    plt.legend()

    # Show the plot on the screen
    plt.show()
    
    mpl_fig = plt.gcf()
    plotly_fig = tls.mpl_to_plotly(mpl_fig)
    st.plotly_chart(plotly_fig)

    # Extract the last 'prediction_days' values from the model inputs
    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]

    # Convert the real_data list to a numpy array
    real_data = np.array(real_data)

    # Reshape real_data to a 3D array with the appropriate dimensions for the LSTM model
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    # Generate a prediction using the LSTM model with the real_data input
    prediction = lstm_model.predict(real_data)

    # Invert the scaling applied to the prediction to obtain the actual value
    prediction = scaler.inverse_transform(prediction)

    # Print the most recent stock price
    stock_data = yf.download(company, period='1d')
    most_recent_close = stock_data['Close'][-1]
    print(f"The most recent closing price for {company} is ${most_recent_close}")

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


with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Individual S&P 500 Stock Metrics', 'Glossary and Explanations'],
        icons=['house', 'graph-up-arrow', 'question'], menu_icon="cast", default_index=1)

if selected == "Individual S&P 500 Stock Metrics":
    st.title("Individual S&P 500 Stock Metrics")
    st.sidebar.header('User Input Parameters')
    symbol, start, end, buying_price, balance = basic_user_input_features()

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    # Read data
    data = yf.download(symbol, start, end)
    data.columns = map(str.lower, data.columns)
    df = data.copy()
    df = ta.add_all_ta_features(df, "open", "high", "low", "close", "volume", fillna=True)
    df_trends = df[['close', 'trend_sma_fast', 'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow', ]]
    df_momentum = df[
        ['momentum_rsi', 'momentum_roc', 'momentum_tsi', 'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal',
         'momentum_wr', 'momentum_ao', 'momentum_kama']]

    # Price
    daily_price = data.close.iloc[-1]
    portfolio = daily_price * balance

    st.title(f"{symbol} :dollar:")

    st.header(f"{symbol}'s Previous Week Performance")
    st.dataframe(data.tail())
    st.header("Today's value of " + f"{symbol}")

    st.markdown(f'Daily {symbol} price: {daily_price}')

    st.markdown(f'{symbol} price per quantity: {portfolio}')

    st.dataframe(data.tail(1))

    st.header(f"Candlestick for {symbol}")
    # Initialize figure
    fig = go.Figure()
    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df.open,
                                 high=df.high,
                                 low=df.low,
                                 close=df.close,
                                 visible=True,
                                 name='Candlestick', ))

    fig.add_shape(
        # Line Horizontal
        type="line",
        x0=start,
        y0=buying_price,
        x1=end,
        y1=buying_price,
        line=dict(
            color="black",
            width=1.5,
            dash="dash",
        ),
        visible=True,
    )
    for column in df_trends.columns.to_list():
        fig.add_trace(
            go.Scatter(x=df_trends.index, y=df_trends[column], name=column, ))
    fig.update_layout(height=800, width=1000, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

    st.header(f"Trends for {symbol}")
    fig = go.Figure()
    for column in df_trends.columns.to_list():
        fig.add_trace(
            go.Scatter(x=df_trends.index, y=df_trends[column], name=column, ))
    # Adapt buttons start
    button_all = dict(label='All', method='update', args=[
        {'visible': df_trends.columns.isin(df_trends.columns), 'title': 'All', 'showlegend': True, }])


    def create_layout_button(column):
        return dict(label=column,
                    method='update',
                    args=[{'visible': df_trends.columns.isin([column]),
                           'title': column,
                           'showlegend': True,
                           }])


    fig.update_layout(updatemenus=[go.layout.Updatemenu(active=0, buttons=([button_all]) + list(
        df_trends.columns.map(lambda column: create_layout_button(column))))], )
    # Adapt buttons end
    # add slider
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
            type="date"
        ))
    fig.update_layout(height=800, width=1000, updatemenus=[
        dict(direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0, xanchor="left", y=1.15,
             yanchor="top", )], )
    # Candlestick
    st.plotly_chart(fig)

    # momentum indicators
    st.header(f"Momentum Indicators for {symbol}")
    trace = []
    Headers = df_momentum.columns.values.tolist()
    for i in range(9):
        trace.append(go.Scatter(x=df_momentum.index, name=Headers[i], y=df_momentum[Headers[i]]))
    fig = make_subplots(rows=9, cols=1)
    for i in range(9):
        fig.append_trace(trace[i], i + 1, 1)
    fig.update_layout(height=2200, width=1000)
    st.plotly_chart(fig)

    st.header(f"LSTM Chart for {symbol}")
    provide_LSTM_model(symbol, period='3mo')
    st.header(f"Stock Grades for {symbol}")

if selected == "Glossary and Explanations":
    st.title("Glossary and Explanations test")
if selected == "Home":
    st.title("Home Dashboard")
