import os
from joblib import load
import json

def load_num_model():
    path = os.path.join(os.getcwd(), "saved_model_params", "lr_model.joblib")
    return load(path)

def load_raw_df():
    import pandas as pd
    
    path = os.path.join(os.getcwd(), "csv_files", "inference", "historical_data.csv")
    return pd.read_csv(path, index_col="Date", parse_dates=True), path

def load_num_config():
    path = os.path.join(os.getcwd(), "saved_model_params", "json_files", "num_config.json")
    
    with open(path, "r") as file:
        return json.load(file)
    
def load_nlp_config():
    path = os.path.join(os.getcwd(), "saved_model_params", "json_files", "nlp_config.json")
    
    with open(path, "r") as file:
        return json.load(file)