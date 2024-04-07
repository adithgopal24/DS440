# Import packages
import yfinance as yf
import pandas as pd

# Read and print the stock tickers that make up S&P500
tickers = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
print(tickers.Symbol)

COMPANY = 'TSLA'
if (tickers.Symbol == COMPANY).any().any():
    print('valid')
else:
    print('not valid')