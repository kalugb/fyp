from flask import send_from_directory, Blueprint, session
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

base_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
dataset_dir = os.path.join(base_dir, "csv_files", "inference")

download_app = Blueprint("download", __name__)

@download_app.route("/download_historical/<stock_symbol>")
def download_historical(stock_symbol):    
    return send_from_directory(
        directory=dataset_dir,
        path=f"{stock_symbol}.csv",
        as_attachment=True, 
        download_name=f"{stock_symbol}.csv"
    )