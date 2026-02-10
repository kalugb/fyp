from flask import Flask, render_template, request, jsonify, session, url_for, send_from_directory
import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from website.backend_functions import update_dataset, generate_session_key, get_exchange_rate
from admin_backend import admin_app
from download_backend import download_app

df_inference_path = os.path.join(os.getcwd(), "csv_files", "inference", "historical_data.csv")
raw_df = pd.read_csv(df_inference_path, index_col="Date", parse_dates=True)
raw_df = update_dataset(df_inference_path)

app = Flask(__name__)
app.register_blueprint(admin_app)
app.register_blueprint(download_app)
app.secret_key = generate_session_key()

# get current conversion rate
conversion_rate = get_exchange_rate()

# main_page.html
@app.route("/")
def home():
    return render_template("main_page.html", name="Main Page", conversion_rate=f"{conversion_rate:.2f}")

# retrieve latest closing price
@app.route("/retrieve_data", methods=["POST"])
def retrieve_data():
    data = raw_df.iloc[-1, :-1]

    # data = np.array([adj close]), so that's why the below 
    closing_price_usd = data["Adj Close"]
    closing_price_myr = closing_price_usd * conversion_rate

    return jsonify({"data_usd": closing_price_usd,
                    "data_myr": closing_price_myr})

@app.route("/convert_to_usd", methods=["POST"])
def convert_to_usd():
    price_myr = request.form.get("stockPriceMYR")
    
    return jsonify({"data": float(price_myr) / conversion_rate})

@app.route("/convert_to_myr", methods=["POST"])
def convert_to_myr():
    price_usd = request.form.get("stockPriceUSD")
    
    return jsonify({"data": float(price_usd) * conversion_rate})

@app.route("/get_data", methods=["POST"])
def get_data():
    stock_myr = request.form.get("stockPriceMYR")
    stock_usd = request.form.get("stockPriceUSD")
    news_headline = request.form.get("newsHeadline")
    
    try:
        if not stock_myr and not stock_usd:
            error = "Please enter at least one field (MYR or USD)"
            return render_template("main_page.html", error=error,
                                stock_myr=stock_myr, 
                                stock_usd=stock_usd,
                                conversion_rate=f"{conversion_rate:.2f}",
                                news_headline=news_headline,
                                show_error=True)
        else:
            stock_myr = float(stock_myr)
            stock_usd = float(stock_usd)
            
        if stock_myr < 0 or stock_usd < 0:
            error = "Please enter a positive value for closing price."
            return render_template("main_page.html", error=error,
                                stock_myr=stock_myr, 
                                stock_usd=stock_usd,
                                conversion_rate=f"{conversion_rate:.2f}",
                                news_headline=news_headline,
                                show_error=True)

    except ValueError:
        error = "Please enter a valid number (2 significant number) for the closing price"
        return render_template("main_page.html", error=error,
                                stock_myr=stock_myr, 
                                stock_usd=stock_usd,
                                conversion_rate=f"{conversion_rate:.2f}",
                                news_headline=news_headline,
                                show_error=True)

    advice_msg = ""
    if not news_headline:
        news_headline = "(No news headline given)"
        advice_msg = "Tips: It is better to input a news headline for <b>better and more robust prediction</b>."
    
    return render_template("confirmation.html", 
                           stock_myr="{:.2f}".format(stock_myr), 
                           stock_usd="{:.2f}".format(stock_usd),
                           news_headline=news_headline,
                           advice_msg=advice_msg)

# waiting_page.html (from confirmation.html)
@app.route("/waiting_page", methods=["POST"])
def waiting_page():
    from website.backend_functions import mean_reversion
    
    price_usd = float(request.form.get("priceUSD"))
    text = request.form.get("textSentiment", "")
    text = text if text != "(No news headline given)" else ""
    
    if price_usd == 0 or price_usd == None:
        error = "Input cannot be None. Please return to main page and input a valid closing price"
        return render_template("confirmation.html", error=error, price_usd=price_usd, text=text)
    
    df = raw_df.copy()
    
    if df.loc[df.index[-1], "Adj Close"] != price_usd:
        df.loc[df.index[-1], "Adj Close"] = price_usd
        
    df = mean_reversion(df)
    data = df.iloc[-1, :-1]
    
    num_data = {"adj_close": data.iloc[0], "returns": data.iloc[1], "ma": data.iloc[2], "ratio": data.iloc[3]}
    nlp_data = {"text": str(text)}
    
    return render_template("waiting_page.html", num_data=num_data, nlp_data=nlp_data)

# get result, user at waiting page
@app.route("/run_model", methods=["POST"])
def run_model():
    from shared_utils.num_inference import predict_position
    
    data = request.get_json()
    num_data = data["num_data"]
    nlp_data = data["nlp_data"]

    num_result, num_label, _ = predict_position(**num_data)
    
    if nlp_data["text"] != "":
        from shared_utils.nlp_inference import predict_sentiment
        nlp_result, nlp_label, _ = predict_sentiment(**nlp_data)
    else:
        nlp_result, nlp_label = "No news headline given", 0
        
    # debugging purpose
    # import time
    # time.sleep(100000)
    
    session["result"] = {"num": [num_result, str(num_label)], "nlp": [nlp_result, str(nlp_label)]}
    
    return jsonify(redirect=url_for("result"))

# result fetched, user at result page
@app.route("/result")
def result():
    data = session["result"]
    
    num_result, num_label = data["num"]
    nlp_result, nlp_label = data["nlp"]
    
    overall_label = int(num_label) + int(nlp_label)
    overall_result = ""
    
    if overall_label == -2:
        overall_result = "Strong sell position"
    elif overall_label == -1:
        overall_result = "Sell position"
    elif overall_label == 0:
        overall_result = "Hold position"
    elif overall_label == 1:
        overall_result = "Buy position"
    elif overall_label == 2:
        overall_result = "Strong buy position"
    else:
        print("Undefined results")
        overall_result = "Undefined"
                
    return render_template("result.html",
                           num_result=num_result, 
                           nlp_result=nlp_result,
                           overall_result=overall_result)
    
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
    
    
    