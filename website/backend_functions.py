# might be replaced by database instead
from datetime import date, timedelta, datetime
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np

raw_inference_path = os.path.join(os.getcwd(), "csv_files", "inference", "historical_data.csv")
cleaned_inference_path = os.path.join(os.getcwd(), "csv_files", "inference", "cleaned_data.csv")

def generate_session_key():
    import string
    import secrets
    
    character_list = string.ascii_letters + string.digits
    session_key_length = 32 + secrets.randbelow(len(character_list) - 32)
    session_key = "".join(secrets.choice(character_list) for _ in range(session_key_length))
    
    return str(session_key)

def get_last_trading_day():
    today = datetime.today() - timedelta(hours=20.5)
    
    if today.weekday() == 5:
        last_trading_day = today - timedelta(days=1)
    elif today.weekday() == 6:
        last_trading_day = today - timedelta(days=2)
    else:
        last_trading_day = today
        
    return last_trading_day.strftime("%Y-%m-%d")

def get_dataset():
    import yfinance as yf
    
    ticker_sym = "AAPL"

    ticker = yf.Ticker(ticker_sym)

    df = ticker.history(period="30y", auto_adjust=False)
    
    df = df.drop(columns=["Dividends", "Stock Splits"])
    
    print("Historical data generated successfully")
    
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
    
def update_dataset(df_path):
    retrieved_raw = False
    
    if not os.path.exists(df_path):
        raw_df = get_dataset()
        retrieved_raw = True
        retrieved_latest = True
    else:
        raw_df = pd.read_csv(df_path, index_col="Date", parse_dates=True)
    
    latest_date_in_csv = str(raw_df.index[-1].date())
    retrieved_latest = False
    
    last_trading_day = get_last_trading_day()
    
    if latest_date_in_csv != last_trading_day and not retrieved_raw:
        print("Retrieving lateset stock data...")
        
        raw_df = get_dataset()
        
        print(f"Latest dataset retrieved with latest date: {str(raw_df.index[-1].date())}")
        retrieved_latest = True
    else:
        print("Using the current dataset")
        
    # update
    if retrieved_latest or retrieved_raw:
        raw_df.to_csv(df_path, index_label="Date", index=True)
        df_cleaned = raw_df.dropna()
        df_cleaned = mean_reversion(df_cleaned)
        df_cleaned.to_csv(cleaned_inference_path, index_label="Date", index=True)
        print("Latest dataset has been updated")
        
    return raw_df
    
if __name__ == "__main__":
    raw_df = update_dataset(raw_inference_path)
    
    