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
    buying_price = st.sidebar.number_input("Buying Price (USD)", value=0.2000, step=0.0001)
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
    periods = ['3mo', '6mo', '1y', 'ytd']
    single_period = st.sidebar.radio('Select a Time Range for LSTM analysis:', periods)
    return single_period
def load_data(company, period):
    msft = yf.Ticker(company)
    hist = msft.history(period)
    hist.reset_index('Date', inplace=True)
    hist.insert(0, 'Name', company)
    hist.drop(columns=['Dividends', 'Stock Splits'])
    hist = hist[['Name', 'Date', 'Open', 'Close', 'High', 'Low', 'Volume']]
    return hist


def provide_LSTM_model(company, period):
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
        filepath='weights_best.hdf5.keras',
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


# Define normalization function
def normalize(value, min_value, max_value):
    """ Normalizes a value to a scale between 0 and 1, capping it at 1 """
    normalized_value = (value - min_value) / (max_value - min_value)
    return min(normalized_value, 1)  # Cap at 1


# Define function to combine P/E ratio and RSI scores
def combine_pe_rsi_grade(pe_ratio, rsi, pe_weight, rsi_weight, pe_min, pe_max, rsi_min, rsi_max):
    """ Combines normalized P/E ratio and RSI into a weighted score, capping the combined score at 1 """
    normalized_pe = normalize(pe_ratio, pe_min, pe_max)
    normalized_rsi = normalize(rsi, rsi_min, rsi_max)

    combined_score = (pe_weight * normalized_pe) + (rsi_weight * normalized_rsi)
    return min(combined_score, 1)  # Cap at 1


# Function to compute RSI
def compute_RSI(data, window=14):
    """ Computes the Relative Strength Index (RSI) for given data """
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    eq = 100 - (100 / (1 + rs))
    return eq

def compute_most_popular_stock_scores():
    popular_tickers = ['MSFT', 'AAPL', 'TSLA', 'META']
    yf_data = yf.download(popular_tickers, start="2019-01-01", end="2023-12-31")
    df_scores = pd.DataFrame(index=popular_tickers)
    # Initialize minimum and maximum values for PE and RSI for normalization
    pe_min, pe_max, rsi_min, rsi_max = 10, 20, 30, 70  # Adjust based on historical data

    # Weights for P/E and RSI
    pe_weight, rsi_weight = 0.5, 0.5

    # Calculate the indicators and scores
    for ticker in popular_tickers:
        stock = yf.Ticker(ticker)
        # Get stock financial information
        pe_ratio = stock.info.get('trailingPE', np.nan)
        rsi = compute_RSI(yf_data['Adj Close'][ticker]).iloc[-1]

        # Get the combined score for P/E ratio and RSI
        df_scores.loc[ticker, 'Stock Grade'] = combine_pe_rsi_grade(pe_ratio, rsi, pe_weight, rsi_weight, pe_min,
                                                                       pe_max,
                                                                       rsi_min, rsi_max)
    # Apply some criteria to assign a stock grade based on the combined score
    df_scores['Stock Action Label'] = pd.cut(df_scores['Stock Grade'],
                                      bins=[0, 0.2, 0.4, 0.6, 0.8, float('inf')],
                                      labels=['Potential Strong Sell', 'Potential Sell', 'Potential Hold',
                                              'Potential Buy', 'Potential Strong Buy'],
                                      include_lowest=True)

    # Display the DataFrame with combined scores and stock grades
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    return df_scores[['Stock Grade', 'Stock Action Label']]

def compute_single_stock_score(input):
    yf_data = yf.download(input, start="2019-01-01", end="2023-12-31")
    #input_type = [{input}]
    df_scores = pd.DataFrame(index=[input])
    # Initialize minimum and maximum values for PE and RSI for normalization
    pe_min, pe_max, rsi_min, rsi_max = 10, 20, 30, 70  # Adjust based on historical data

    # Weights for P/E and RSI
    pe_weight, rsi_weight = 0.5, 0.5

    stock = yf.Ticker(input)
    # Get stock financial information
    pe_ratio = stock.info.get('trailingPE', np.nan)
    rsi = compute_RSI(yf_data['Adj Close']).iloc[-1]

    # Get the combined score for P/E ratio and RSI
    df_scores.loc[input, 'Stock Grade'] = combine_pe_rsi_grade(pe_ratio, rsi, pe_weight, rsi_weight, pe_min,
                                                                   pe_max,
                                                                   rsi_min, rsi_max)
    # Apply some criteria to assign a stock grade based on the combined score
    df_scores['Stock Action Labels'] = pd.cut(df_scores['Stock Grade'],
                                      bins=[0, 0.2, 0.4, 0.6, 0.8, float('inf')],
                                      labels=['Potential Strong Sell', 'Potential Sell', 'Potential Hold',
                                              'Potential Buy', 'Potential Strong Buy'],
                                      include_lowest=True)

    # Display the DataFrame with combined scores and stock grades
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    return df_scores[['Stock Grade', 'Stock Action Labels']]

def compute_custom_stock_score(pe_ratio, rsi):
    pe_min, pe_max, rsi_min, rsi_max = 10, 20, 30, 70  # Adjust based on historical data
    pe_weight, rsi_weight = 0.5, 0.5
    stock_grade = combine_pe_rsi_grade(pe_ratio, rsi, pe_weight, rsi_weight, pe_min,
                                                               pe_max,
                                                               rsi_min, rsi_max)
    label = ''
    if stock_grade <= 0.2:
        label = 'Potential Strong Sell'
    elif stock_grade <= 0.4:
        label = 'Potential Sell'
    elif stock_grade <= 0.6:
        label = 'Potential Hold'
    elif stock_grade <= 0.8:
        label = 'Potential Buy'
    else:
        label = 'Potential Strong Buy'

    output_1 = f"Stock Grade: {stock_grade}"
    output_2 = f"Stock Action Label: {label}"
    return output_1, output_2

with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Individual S&P 500 Stock Metrics', 'Definitions and Explanations'],
        icons=['house', 'graph-up-arrow', 'question'], menu_icon="cast", default_index=1)


if selected == "Individual S&P 500 Stock Metrics":
    st.title("Individual S&P 500 Stock Metrics")
    st.sidebar.header('Basic User Input Parameters')
    symbol, start, end, buying_price, balance = basic_user_input_features()
    st.sidebar.header('LSTM Time Period')
    period = advanced_user_input_features()


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

    fig.update_layout(height=800, width=1000, xaxis_rangeslider_visible=False, margin=dict(l=50, r=50, t=50, b=50),
                      yaxis_title = 'USD')
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
    provide_LSTM_model(symbol, period=period)
    st.header(f"Stock Grades for {symbol}")
    string_symbol = str(symbol)
    st.table(compute_single_stock_score(string_symbol))

elif selected == "Definitions and Explanations":
    st.title("Definitions and Explanations")
    st.write("Utilize our own Defintions and Explanations Page to learn more about the S&P 500 market and all sorts of jargon, metrics, and formulas used on our website!")
    st.header("Definitions")
    st.write("""**P/E Ratio**: The price-to-earnings ratio is a widely used metric in the stock market that measures a company’s current stock price relative to its earnings per share. Stocks with higher P/E ratios are often perceived as possibly overvalued, as its stock price is high compared to their earnings. Conversely, investors might look for stocks with lower P/E ratios as potential buying opportunities, as they could be seen as undervalued. """)
    st.write("""**RSI**: The Relative Strength Index is a technical analysis indicator used to measure the speed and change of price movements. It is intended to identify whether a stock is potentially overbought or oversold, suggesting a possible reversal point in the stock’s price. The RSI is typically displayed as an oscillator and can have a reading between 0 to 100. Normally, an RSI reading above 70 suggests that a stock may be overbought, indicating a potential sell. And an RSI reading below 30 indicates that a stock may be oversold. If a stock’s price is making new highs but the RSI is not, it may indicate a lack of momentum and a potential reversal""")
    st.write("""**SMA**: The Simple Moving Average is a technical analysis tool used to smooth out price data by creating a constantly updated average price. The average can be computed over a specific period of days, such as 20 or 50 days. When a stock's price is above its SMA, it often indicates an upward trend. Conversely, if the price is below its SMA, this typically signals a downward trend.""")
    st.write("""**EMA**: The Exponential Moving Average is a type of moving average used in technical analysis to measure the trends over a set period. The EMA differs from SMA in that it places a greater emphasis on recent price data. If the price is above the EMA, it is generally considered to be in an uptrend, suggesting buying opportunities. If the price is below the EMA, it is often considered to be in a downtrend, indicating potential selling or short-selling opportunities.""")
    st.write("""**ROC**: Representing Rate of Change, this indicator measures the percentage change in price between the current price and the price a certain number of periods ago. The ROC is used to identify momentum or trend reversals.
As a momentum indicator, it reflects the velocity of price changes. 
If ROC is rising, it indicates an increase in momentum, suggesting that prices are moving up at a faster rate. Conversely, if ROC is falling, it means the momentum is decreasing and prices might start dropping or stabilizing.
""")
    st.write("""**TSI**: The True Strength Index is a momentum oscillator that helps identify trends and reversals. It combines moving averages of the underlying asset’s price momentum, typically smoothing price movements to filter out noise and better highlight trends.
It is particularly useful in identifying overbought and oversold conditions. 
A high positive TSI indicates strong upward momentum, while a high negative TSI suggests strong downward momentum. The crossing of the TSI line over its signal line can indicate potential buy or sell opportunities based on changes in momentum.
""")
    st.write("""**UO**: The Ultimate Oscillator, developed by Larry Williams, combines short, medium, and long-term market trends into one value. It aims to reduce volatility and false signaling typical in many oscillators considering only one time frame.
This indicator combines multiple time frames, which helps smooth out the signals for momentum.
When the UO rises above certain threshold levels, it suggests increasing buying momentum. Conversely, falling below certain levels indicates increasing selling momentum. Divergences between the UO and price action can also indicate potential reversals.
""")
    st.write("""**Stoch**: Stoch compares an asset's closing price to a range of its prices over a certain period. It is used to generate overbought and oversold trading signals.
It is used to predict price turning points by comparing a commodity's closing price to its price range over a given time period.""")
    st.write("""**Stoch Signal**: Often, Stoch Signal refers to the signal line in the stochastic oscillator. The signal line is a moving average of the stochastic oscillator value itself and is used to interpret buy and sell signals more clearly.
The signal line in the stochastic oscillator, usually a moving average of the stochastic values, helps confirm momentum shifts.
""")
    st.write("""**WR**: The Williams %R is a momentum indicator that measures overbought and oversold levels. It is similar to the stochastic oscillator but inverted. This indicator compares a stock's closing price to the high-low range over a specific period, typically 14 days. This indicator is similar to the stochastic oscillator and is particularly useful for spotting overbought and oversold levels.""")
    st.write("""**AO**: Developed by Bill Williams, the Awesome Oscillator calculates the difference between a 34 period and a 5 period simple moving averages. The averages are based on the midpoints of the bars (H+L)/2 instead of closing prices. The AO is used to gauge market momentum.
This indicator captures market momentum by calculating the difference between the moving averages over two different periods. 
""")
    st.write("""**KAMA**: Kaufman's Adaptive Moving Average was developed by Perry Kaufman and adapts to the price movement of an asset in an attempt to reduce noise and highlight significant trends more clearly. It adjusts its responsiveness based on the volatility of the prices.
Unlike standard moving averages, KAMA adjusts its sensitivity to price movements.
""")

    st.header("What data was used?")
    st.write("Our dataset was obtained from Yahoo Finance, utilizing the Yahoo Finance API in Python to retrieve the latest data available for all S&P 500 stocks, updating every business day (M-F). A historical dataset, from 2019 to the current day, was utilized for our models.")

    st.header("How did we produce our Stock Grades? What do these Grades mean?")
    st.write("StockGrader.io calculates a composite, numeric Stock Grade for each stock by analyzing and combining the stock's P/E ratio and RSI. These two indicators are used because these two values ​​are the two most important factors determining a stock’s performance, and their weighted distribution is equal in our formula. The process of developing a Stock Grade utilizes data standardization, weight allocation, and composite score calculations to provide investors with stock assessments based on financial and market momentum indicators. Our Stock Grades are designed to help investors easily understand how a stock is currently performing, at the moment.")
    st.write(" ")
    st.write("Along with a Stock Grade, each stock will have a “Stock Action Label”, which indicates what a long-term investor should likely do with the respective stock at the current moment. The labels range from “Potential Strong Sell”, up to “Potential Strong Buy”, indicating if a person should sell, hold, or buy a stock at the moment.")
    st.write(" ")
    st.write("*Note:* Certain stocks may not have a Stock Grade for the time being due to unavailable P/E Ratio or RSI metrics. This is shown in the “Individual S&P 500 Stock Metrics page, where a stock’s Stock Grade may be shown as “N/A”.")

    st.header("What is LSTM? How did we utilize it to predict stock prices?")
    st.write("LSTM (Long short-term memory) is a machine learning model that utilizes recurrent neural networks. LSTMs predict stock prices by learning from historical data, utilizing their unique gates to manage information flow and retain relevant patterns over time. They adjust their weights based on prediction errors, improving accuracy with more data. This ability to learn and remember long-term dependencies in stock price trends, while dynamically adapting to new information, makes LSTMs effective for financial forecasting. We decided to use an LSTM model to predict stock prices because of its ability to handle multi-input data with time steps")
    st.write(" ")
    st.write("**Why use LSTMs?**")
    st.write("LSTMs excel in stock price prediction due to their ability to remember information for long periods and handle the complexities of time-series data, like stock prices. Their architecture allows them to capture long-term dependencies and avoid the vanishing gradient problem, making them superior to traditional models for understanding trends and patterns that span various time frames.")
    st.write("**How do LSTMs work?**")
    st.write("LSTMs predict stock prices by learning from historical data, utilizing their unique gates to manage information flow and retain relevant patterns over time. They adjust their weights based on prediction errors, improving accuracy with more data. This ability to learn and remember long-term dependencies in stock price trends, while dynamically adapting to new information, makes LSTMs effective for financial forecasting.")
    st.write("**What are the strengths of LSTMs?**")
    st.write("The primary strengths of LSTM models include their long-term memory capability, flexibility in handling sequences of variable lengths, and robustness to long gaps in data. These features make LSTMs particularly suited for applications like stock price prediction, where understanding long-term patterns and dealing with time-series data are crucial for accurate forecasting.")
    st.write("**What are the weaknesses of LSTMs?**")
    st.write("LSTMs come with challenges, such as their computational complexity, which requires significant resources and time for training. They also have a higher risk of overfitting due to their ability to learn detailed patterns, and finding the optimal set of hyperparameters can be difficult, making the model-tuning process challenging.")

    st.header("How do you read a Candlestick chart?")
    st.write("For each stock, we have produced a Candlestick chart that shows a stock's opening and closing prices, along with its highest and lowest price of the day. Each vertical bar may be colored green if the stock's closing price was higher than its opening price, or may be colored red if the stock's closing price was lower than its opening price.")
    st.write("At the top/bottom of a vertical bar, there may be a line sticking out of it, called a Shadow. If a Shadow is coming out of the top of a vertical bar, it is called an Upper Shadow, while a line coming out of the bottom of the vertical bar is called a Lower Shadow.")
    st.write("A short upper shadow on a green bar indicates the closing price was similar to the highest price of the day. ")
    st.write("The visualizations of a Candlestick chart can be utilized by an investor to see how a stock has been performing, with emphasis on the colors and size of the bars to help an investor see its pricing behavior easier.")
    st.write("For example, using a Candlestick chart, it is easier to see if a stock's price will fall (Bearish behavior) or rise over time (Bullish behavior). Generally, a stock can be seen as Bearish if there is a large red bar that is directly to the right of a much smaller green bar. This indicates the stock's price may continue to fall, so an investor should sell the stock.")
    st.image('Bearish.png', caption='A Bearish Candlestick chart')
    st.write("The opposite is true for Bullish stocks, where a large green bar is directly to the right of a much smaller red bar. This indicuates that a stock's price may continue to increase, so an investor should buy the stock.")
    st.image('Bullish.png', caption='A Bullish Candlestick chart')
    st.write("For more information about Candlestick charts, see this [link](https://www.investopedia.com/trading/candlestick-charting-what-is-it/)")

    st.header("How can the Trends charts be used?")
    st.write("StockGrader.io's Trend charts can be used to complement the provided Candlestick chart for a particular stock. The Fast SMA trend value is one with generally a smaller period of time, where the moving average is generally calculated between 10-50 days. The Slow SMA trend value generally calulates moving averages from 50-200 days. People looking to find more recent price changes/trends utilize Fast SMA, while long-term traders look to utilize Slow SMA, since it is can find long-term trends, and isn't as sensitive to quick, short spikes or drops of price. ")
    st.write("Similar to SMA, the Fast EMA trend value utilizes a shorter period of time, generally about 12-26 periods. Slow EMA utilizes anywhere from 50-200 periods. Slow EMA generally provides more weight to older price data, compared to new ones. This makes Slow EMA more stable long-term, but does not pick up on quick spikes/drops of price. Like SMA, long-term investors would more likly utilize Slow EMA compared to Fast EMA. ")

    st.header("How can the Momentum charts be used?")
    st.write("StockGrader.io has created numerous Momentum Charts using a variety of market-performance variables, such as SO and ROC, among others. Each of these momentum indicators can be used to see how a stock is trending, and if the trend is weak or strong. Many of these variables utilize both current and historic prices to produce their indicators.  ")


elif selected == "Home":
    st.title("Home Dashboard")
    st.subheader("Popular S&P 500 Stock Grades")
    most_popular_scores = compute_most_popular_stock_scores()
    st.table(most_popular_scores)
    st.subheader("Create your own Stock Grade below: ")
    try:
        input_1 = int(st.text_input("Insert a P/E Ratio between the values of 10-20:"))
        input_2 = int(st.text_input("Insert an RSI between the values of 30-70:"))
        if not input_1:
            print(st.warning("Please enter a number for P/E Ratio"))
        if not input_2:
            print(st.warning("Please enter a number for RSI"))
        if input_1 < 10 or input_1 > 20:
            print(st.warning("Please enter a number between 10 and 20 for the P/E Ratio"))
        if input_2 < 30 or input_2 > 70:
            print(st.warning("Please enter a number between 30 and 70 for the RSI"))
        submit = st.button('Generate Stock Grade')
        st.subheader("Created Stock Grade:")
        if submit:
            st.markdown(compute_custom_stock_score(input_1, input_2))
    except:
        # Prevent the error from propagating into your Streamlit app.
        pass

    #label_visibility = st.session_state.visibility,
    #disabled = st.session_state.disabled,
    #placeholder = st.session_state.placeholder