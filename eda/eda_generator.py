import pandas as pd
from ydata_profiling import ProfileReport
import os

working_dir = os.getcwd()
csv_dir = os.path.join(working_dir, "csv_files")
current_dir = os.path.dirname(__file__)

df_raw_path = os.path.join(csv_dir, "historical_stock_data.csv")
df_raw = pd.read_csv(df_raw_path, index_col=0)

df_cleaned_path = os.path.join(csv_dir, "cleaned_historical_stock_data.csv")
df_cleaned = pd.read_csv(df_cleaned_path, index_col=0)

def generate_eda_report(df, save_path, title):
    profile = ProfileReport(df, title=f"EDA for {title}")

    profile.to_file(save_path)

    print("HTML created successfully")

raw_eda_save_path = os.path.join(current_dir, "raw_eda.html")
generate_eda_report(df_raw, raw_eda_save_path, "raw dataset")

cleaned_eda_save_path = os.path.join(current_dir, "cleaned_eda.html")
generate_eda_report(df_cleaned, cleaned_eda_save_path, "cleaned dataset")
        

