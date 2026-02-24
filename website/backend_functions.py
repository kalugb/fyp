# might be replaced by database instead
from datetime import date, timedelta, datetime
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np

def generate_session_key():
    import string
    import secrets
    
    character_list = string.ascii_letters + string.digits
    session_key_length = 32 + secrets.randbelow(len(character_list) - 32)
    session_key = "".join(secrets.choice(character_list) for _ in range(session_key_length))
    
    return str(session_key)

def get_exchange_rate():
    from forex_python.converter import CurrencyRates
    c = CurrencyRates()
    
    rate = c.get_rate("USD", "MYR")
    
    return rate

def get_raw_path() -> dict:
    stock_list = ["AAPL", "MSFT", "BA", "AMZN"]
    paths = []
    stock_path_list = {}
    
    for stock in stock_list: 
        path = os.path.join(os.getcwd(), "csv_files", "inference", f"{stock}.csv")
        paths.append(path)
        
    for stock, path in zip(["AAPL", "MSFT", "BA", "AMZN"], paths):
        stock_path_list[stock] = path
        
    return stock_path_list

def get_raw_df(path) -> pd.DataFrame:
    import pandas as pd
    
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    
    return df

def get_last_trading_day():
    today = datetime.today() - timedelta(hours=20.5)
    
    if today.weekday() == 5:
        last_trading_day = today - timedelta(days=1)
    elif today.weekday() == 6:
        last_trading_day = today - timedelta(days=2)
    else:
        last_trading_day = today
        
    return last_trading_day.strftime("%Y-%m-%d")

def get_dataset(ticker_symbol, path) -> pd.DataFrame:
    import yfinance as yf

    ticker = yf.Ticker(ticker_symbol)

    df = ticker.history(period="30y", auto_adjust=False)
    
    df = df.drop(columns=["Dividends", "Stock Splits"])
    
    print("Historical data generated successfully")
    
    df.to_csv(path, index=["Date"])
    
    return df

def mean_reversion(df):
    ma = 21
    df["returns"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    df["ma"] = df["Adj Close"].rolling(ma, center=False).mean()
    df["ratio"] = df["Adj Close"] / df["ma"]

    percentiles_list = [20, 80]
    percentiles = np.percentile(df["ratio"].dropna(), percentiles_list)
    short = percentiles[-1]
    long = percentiles[0]

    df["position"] = 0
    df["position"] = np.where(df["ratio"] < long, 1, df["position"])
    df["position"] = np.where(df["ratio"] > short, -1, df["position"])
    df["position"] = df["position"].shift(-1)

    df = df[["Adj Close", "returns", "ma", "ratio", "position"]]
    
    return df
    
# this runs everything
def update_dataset(stock, df_path):
    retrieved_raw = False
    
    if not os.path.exists(df_path):
        raw_df = get_dataset(stock, df_path)
        retrieved_raw = True
    else:
        raw_df = pd.read_csv(df_path, index_col="Date", parse_dates=True)
    
    latest_date_in_csv = str(raw_df.index[-1].date())
    
    last_trading_day = get_last_trading_day()
    
    if latest_date_in_csv != last_trading_day and not retrieved_raw:
        print("Retrieving lateset stock data...")
        
        raw_df = get_dataset(stock, df_path)
        
        print(f"Latest dataset retrieved with latest date: {str(raw_df.index[-1].date())}")
    else:
        print("Using the current dataset")
        
# runs this function when the website first launches, and when stock is changed
def load_df(stock):
    path = os.path.join(os.getcwd(), "csv_files", "inference", f"{stock}.csv")
    
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    
    return df
    
if __name__ == "__main__":
    # raw_df = update_dataset(raw_inference_path)
    
    print(get_exchange_rate())
    
    