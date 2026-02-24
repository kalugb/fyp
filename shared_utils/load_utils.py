import os
from joblib import load
import json

def load_num_model():
    path = os.path.join(os.getcwd(), "saved_model_params", "lr_model.joblib")
    return load(path)

def get_raw_path():
    stock_list = ["AAPL", "MSFT", "BA", "AMZN"]
    paths = []
    
    for stock in stock_list: 
        path = os.path.join(os.getcwd(), "csv_files", "inference", f"{stock}.csv")
        paths.append(path)
        
    return path

def get_raw_df(path):
    import pandas as pd
    
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    
    return df

def load_num_config():
    path = os.path.join(os.getcwd(), "saved_model_params", "json_files", "num_config.json")
    
    with open(path, "r") as file:
        return json.load(file)
    
def load_nlp_config():
    path = os.path.join(os.getcwd(), "saved_model_params", "json_files", "nlp_config.json")
    
    with open(path, "r") as file:
        return json.load(file)