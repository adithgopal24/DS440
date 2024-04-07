import pandas as pd
import yfinance as yf
import ta
from ta import add_all_ta_features
from ta.utils import dropna
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st
from datetime import date
import datetime
from datetime import datetime, date
from streamlit_option_menu import option_menu
import io
yf.pdr_override()

#st.set_option('deprecation.showfileUploaderEncoding', False)

st.write("""
# StockGrader.io: See a Stock's Performance below, using Yahoo Finance data!
""")

st.sidebar.header('User Input Parameters')

time = pd.to_datetime('now')
today_val = date.today()

# Get S&P 500 constituents
sp500 = yf.Tickers("^GSPC")

# Extract tickers and put them in a list
tickers = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers_names = tickers.Symbol.to_list()

with st.sidebar:
    selected = option.menu(
        menu_
    )


def user_input_features():
    #stock_choice = st.radio("Pick a Stock to see its information: ", [tickers_names])
    stock_choice = st.selectbox('Select Stock to See its info', tickers_names)
    #stock_choice = st.radio("Pick a Stock to see its information: ", [names])
    ticker = st.sidebar.text_input("Ticker", stock_choice)
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


symbol, start, end, buying_price, balance = user_input_features()

start = pd.to_datetime(start)
end = pd.to_datetime(end)

# Read data
data = yf.download(symbol,start,end)
data.columns = map(str.lower, data.columns)
df = data.copy()
df = ta.add_all_ta_features(df, "open", "high", "low", "close", "volume", fillna=True)
df_trends = df[['close','trend_sma_fast','trend_sma_slow','trend_ema_fast','trend_ema_slow',]]
df_momentum = df[['momentum_rsi', 'momentum_roc', 'momentum_tsi', 'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr', 'momentum_ao', 'momentum_kama']]



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
                             name='Candlestick',))

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
         visible = True,
)
for column in df_trends.columns.to_list():
    fig.add_trace(
    go.Scatter(x = df_trends.index,y = df_trends[column],name = column,))
fig.update_layout(height=800,width=1000, xaxis_rangeslider_visible=False)
st.plotly_chart(fig)


st.header(f"Trends for {symbol}")
fig = go.Figure()
for column in df_trends.columns.to_list():
    fig.add_trace(
    go.Scatter(x = df_trends.index,y = df_trends[column],name = column,))
# Adapt buttons start
button_all = dict(label = 'All',method = 'update',args = [{'visible': df_trends.columns.isin(df_trends.columns),'title': 'All','showlegend':True,}])
def create_layout_button(column):
    return dict(label = column,
                method = 'update',
                args = [{'visible': df_trends.columns.isin([column]),
                        'title': column,
                        'showlegend': True,
                        }])
fig.update_layout(updatemenus=[go.layout.Updatemenu(active = 0, buttons = ([button_all]) + list(df_trends.columns.map(lambda column: create_layout_button(column))))],)
# Adapt buttons end
# add slider
fig.update_layout(
    xaxis=dict(
        rangeslider=dict(
            visible=True
        ),
        type="date"
    ))
fig.update_layout(height=800,width=1000,updatemenus=[dict(direction="down",pad={"r": 10, "t": 10},showactive=True,x=0,xanchor="left",y=1.15,yanchor="top",)],)
# Candlestick
st.plotly_chart(fig)


# momentum indicators
st.header(f"Momentum Indicators for {symbol}")
trace=[]
Headers = df_momentum.columns.values.tolist()
for i in range(9):
    trace.append(go.Scatter(x=df_momentum.index, name=Headers[i], y=df_momentum[Headers[i]]))
fig = make_subplots(rows=9, cols=1)
for i in range(9):
     fig.append_trace(trace[i],i+1,1)
fig.update_layout(height=2200, width=1000)
st.plotly_chart(fig)

st.header(f"LSTM Chart for {symbol}")

st.header(f"CNN-LSTM Chart for {symbol}")

st.header(f"Stock Grades for {symbol}")
