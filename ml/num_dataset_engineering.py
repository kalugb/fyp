import yfinance as yf
import pandas as pd
import numpy as np
import os
from collections import Counter

working_dir = os.getcwd()
csv_files = os.path.join(working_dir, "csv_files")
raw_dataset_path = os.path.join(csv_files, "raw", "historical_stock_data.csv")
cleaned_dataset_path = os.path.join(csv_files, "cleaned", "cleaned_historical_stock_data.csv")
window = 14

def get_dataset(generate: bool = False):
    if os.path.exists(raw_dataset_path) and not generate:
        print("Raw dataset file exist, using existing file")
        historical_data = pd.read_csv(raw_dataset_path)
    else:
        print("Raw file doens't exist or generate=True, generating one now...")
        ticker_sym = "AAPL"

        ticker = yf.Ticker(ticker_sym)

        historical_data = ticker.history(period="30y", auto_adjust=False)
        
        historical_data = historical_data.drop(columns=["Dividends", "Stock Splits"])
        
        historical_data.to_csv(raw_dataset_path, index=["Date"])
        print("Historical data generated successfully")
    
    return historical_data

def mean_reversion(df: pd.DataFrame):
    # feature engineering, using moving average window = 21
    ma = 21
    df["returns"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    df["ma"] = df["Adj Close"].rolling(ma, center=False).mean()
    df["ratio"] = df["Adj Close"] / df["ma"]

    # setting thresholds for long, short and hold, taking higher top 20% for short, lower top 80% for long
    percentiles_list = [20, 80]
    percentiles = np.percentile(df["ratio"].dropna(), percentiles_list)
    short = percentiles[-1]
    long = percentiles[0]

    df["position"] = 0
    df["position"] = np.where(df["ratio"] < long, 1, df["position"])
    df["position"] = np.where(df["ratio"] > short, -1, df["position"])
    df["position"] = df["position"].shift(-1)

    df = df.dropna().drop_duplicates() if __name__ == "__main__" else df

    df = df[["Adj Close", "returns", "ma", "ratio", "position"]]
    
    return df


if __name__ == "__main__":
    df = get_dataset(generate=False)

    df = mean_reversion(df)

    print(df)
    print(Counter(df["position"]))
    print(df.columns)

    df.to_csv(cleaned_dataset_path, index=["Date"])



