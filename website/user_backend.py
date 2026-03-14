from flask import Flask, render_template, request, jsonify, session, url_for, redirect, flash
import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from website.backend_functions import update_dataset, generate_session_key, get_exchange_rate, get_raw_path, load_df

# default loading to AAPL
stock_path = get_raw_path()
for stock, path in stock_path.items():
    update_dataset(stock, path)
    
raw_df = load_df("AAPL")

# get latest closing price (by default AAPL)
latest_closing_price = raw_df.iloc[-1, :-1]["Adj Close"].tolist()
last_closing_price = raw_df.iloc[-2, :-1]["Adj Close"].tolist()
closing_price_diff = latest_closing_price - last_closing_price
diff_percentage = round((closing_price_diff / last_closing_price) * 100, 2)
if closing_price_diff > 0:
    display_color = "green"
else:
    display_color = "red"
    
previous_close = raw_df.iloc[-2, :-1]["Close"].tolist()
current_open = raw_df.iloc[-1, :-1]["Open"].tolist()
current_volume = int(raw_df.iloc[-1, :]["Volume"].tolist()) 
    
def create_app():
    from admin_backend import admin_app
    from download_backend import download_app
    
    app = Flask(__name__)
    app.secret_key = generate_session_key()
    
    app.register_blueprint(admin_app)
    app.register_blueprint(download_app)

    return app

app = create_app()

# get current conversion rate
conversion_rate = get_exchange_rate()

@app.context_processor
def inject_admin_mode():
    admin_mode = session.get("admin_mode", False)
    stock_active = session.get("stock_active", "AAPL")
    admin_stock_active = session.get("admin_stock_active", "AAPL")
    return dict(admin_mode=admin_mode, stock_active=stock_active, admin_stock_active=admin_stock_active)

# main_page.html
@app.route("/")
def home():
    logout = session.pop("logout", False)
    scroll = session.pop("home_from_about_us", False)
    return render_template("main_page.html", name="Main Page", conversion_rate=conversion_rate, 
                           logout=logout, scroll=scroll, latest_closing_price=latest_closing_price, 
                           display_color=display_color, closing_price_diff=closing_price_diff,
                           diff_percentage=diff_percentage, previous_close=previous_close, 
                           current_open=current_open, current_volume=current_volume)

@app.route("/about_us")
def about_us():
    return render_template("about_us.html")

@app.route("/home_from_about_us")
def home_from_about_us():
    session["home_from_about_us"] = True
    return redirect(url_for("home"))
    
@app.route("/get_stock_data/<symbol>")
def get_stock_data(symbol):    
    using_df = load_df(symbol)
    
    stock_info = {
        "AAPL": ["Apple Inc", "Electronic Technology and Telecommunications Equipment"],
        "MSFT": ["Microsoft Corporation", "Software and Services"],
        "BA": ["The Boeing Company", "Aerospace and Defense"],
        "AMZN": ["Amazon.com, Inc.", "Internet Retail and Services"]
    }
    
    # get latest closing price 
    latest_closing_price = using_df.iloc[-1]["Adj Close"].tolist()
    last_closing_price = using_df.iloc[-2]["Adj Close"].tolist()
    closing_price_diff = latest_closing_price - last_closing_price
    diff_percentage = round((closing_price_diff / last_closing_price) * 100, 2)
    if closing_price_diff > 0:
        display_color = "green"
    else:
        display_color = "red"
        
    previous_close = using_df.iloc[-2, :-1]["Adj Close"].tolist()
    current_open = using_df.iloc[-1, :-1]["Open"].tolist()
    current_volume = int(using_df.iloc[-1, :]["Volume"].tolist())
        
    return jsonify({
        "latest_closing_price": latest_closing_price,
        "closing_price_diff": closing_price_diff,
        "diff_percentage": diff_percentage,
        "display_color": display_color,
        "stock_symbol_name": stock_info[symbol][0],
        "stock_brief_info": stock_info[symbol][1],
        "previous_close": previous_close,
        "current_open": current_open,
        "current_volume": current_volume
    })
    

@app.route("/retrieve_data/<symbol>", methods=["POST"])
def retrieve_data(symbol):
    df = load_df(symbol)
    data = df.iloc[-1, :-1]

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
    
    session["result"] = {"num": [num_result, str(num_label)], "nlp": [nlp_result, str(nlp_label)]}
    session["input"] = {"num": num_data, "nlp": nlp_data}
    
    return jsonify(redirect=url_for("result"))

# result fetched, user at result page
@app.route("/result")
def result():
    data = session["result"]
    
    num_result, num_label = data["num"]
    nlp_result, nlp_label = data["nlp"]
    
    overall_label = int(num_label) + int(nlp_label)
    overall_result = ""
    
    num_input_data = session["input"]["num"]["adj_close"]
    nlp_input_data = session["input"]["nlp"]["text"]
    nlp_input_data = nlp_input_data if nlp_input_data != "" else "(No news headline given)"
    
    overall_result_map = {
        -2: ("Strong Sell Position", "red",
             "Indicators suggest significant downward pressure on the stock price. Thus, suggesting to have strong sell position."),
        -1: ("Sell Position", "red",
             "Signal indicates a mildly downward trend in the stock price. Thus, suggesting to start a sell position."),
        0: ("Hold Position", "#334155",
            "The model detects no clear trend in the current stock price. Thus, suggesting to keep the current position."),
        1: ("Buy Position", "rgb(68, 244, 68)",
            "Signal indicates a midly upward trend in the stock price. Thus, suggesting to start a buy position."),
        2: ("Strong Buy Position", "rgb(68, 244, 68)",
            "Indicators suggest significant upward pressure on the stock price. Thus, suggesting to have strong buy position.")
    }
    
    overall_result, overall_display_color, overall_desc = overall_result_map.get(
        overall_label, ("Undefined", "Undefined", "Model result could not be determined")
    )
        
    # colour display
    num_result_map = {
        -1: ("Sell Position", "red",
             "Signal indicates a mildy downward trend from the moving average."),
        0: ("Hold Position", "#334155",
            "Signal indicates the current prices is around the moving average, thus no clear trend"),
        1: ("Buy Position", "rgb(68, 244, 68)",
            "Signal indicates a midly upward trend from the moving average")
    }
    
    num_result, num_display_color, num_desc = num_result_map.get(
        int(num_label), ("Undefined", "Undefined", "Model result could not be determined")
    )
    
    nlp_result_map = {
        -1: ("Negative Sentiment", "red",
             "The headline reflects negative sentiment toward the company, suggesting a unfavorable market reaction"),
        0: ("Neutral Sentiment", "#334155",
            "The headline appears informational with no strong postive or negative sentiment detected"),
        1: ("Positive Sentiment", "rgb(68, 244, 68)",
            "The headline reflects positive sentiment toward the company, suggesting a favorable market reaction")
    }
    
    if nlp_result != "No news headline given":
        nlp_result, nlp_display_color, nlp_desc = nlp_result_map.get(
            int(nlp_label), ("Undefined", "Undefined", "Model result could not be determined")
        )
    else:
        nlp_result, nlp_display_color, nlp_desc = "No news headline given", "#334155", "No news headline given. Unable to determine sentiment."
                
    return render_template("result.html",
                           num_result=num_result, 
                           nlp_result=nlp_result,
                           overall_result=overall_result,
                           overall_display_color=overall_display_color,
                           overall_desc=overall_desc,
                           num_display_color=num_display_color,
                           num_desc=num_desc,
                           nlp_display_color=nlp_display_color,
                           nlp_desc=nlp_desc,
                           num_input_data=num_input_data,
                           nlp_input_data=nlp_input_data)
        
if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
    
    
