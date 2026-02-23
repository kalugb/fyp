import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, confusion_matrix, classification_report

from joblib import dump

import json

np.random.seed(67)
working_dir = os.getcwd()
saved_param_dir = os.path.join(working_dir, "saved_model_params")

df_path = os.path.join(working_dir, "csv_files", "cleaned", "cleaned_historical_stock_data.csv")
df = pd.read_csv(df_path, index_col=0)

X = df.drop(columns="position")
y = df["position"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=67)

class_weight = {-1: 1.75, 0: 1, 1: 2}
scaler = StandardScaler()
tscv = TimeSeriesSplit(n_splits=5)
lr = LogisticRegression(random_state=67, verbose=0, solver="saga", class_weight=class_weight)

param_grid = {
    "lr__C": [0.8, 1.0, 1.2],
    "lr__max_iter": [1000, 1200, 1500, 1800]
}

pipe = Pipeline([
    ("scaler", scaler),
    ("lr", lr)
])

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=make_scorer(precision_score, zero_division=0, average="macro"),
    cv=tscv,
    refit=True,
    return_train_score=True,
    n_jobs=-1,
    verbose=0
)

grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)

def backtesting(y_pred, X_test):
    # backtesting
    backtest = pd.DataFrame(columns=["position", "pnl"])

    backtest["position"] = y_pred
    backtest["position"] = backtest["position"].shift(1)
    backtest["pnl"] = backtest["position"] * X_test["returns"].values
    backtest = backtest.dropna().reset_index(drop=True)

    # get metrics
    expected_daily_pnl = backtest["pnl"].mean()
    trades_performed = backtest[backtest["position"] != 0] # return dataframe, check df_test at beginning
    win_rate = (trades_performed["pnl"] > 0).astype(int).mean()
    long_frequency = len(backtest.loc[backtest["position"] == 1, "position"]) / len(X_test)
    short_frequency = len(backtest.loc[backtest["position"] == -1, "position"]) / len(X_test)
    total_frequency = long_frequency + short_frequency
    total_trades = int(total_frequency * len(X_test))

    # sharpe ratio without risk-free rate, returns / std
    portfolio_return = expected_daily_pnl
    std = backtest["pnl"].std()
    sharpe = (portfolio_return / std)
    ann_sharpe = (sharpe * np.sqrt(252))

    print(f"Expected daily PnL per share / stock (raw, without cost): {expected_daily_pnl:.4f}")
    print(f"Total Frequency: {total_frequency:.4f}, Long: {long_frequency:.4f}, Short: {short_frequency:.4f}")
    print(f"Sharpe ratio: {sharpe:.4f}, Annualized (using 252 trading days): {ann_sharpe:.4f}")
    print(f"Win rate: {win_rate:.4f}, with total trades performed: {total_trades}")
    
    backtesting_results = {
        "Expected Daily Profit & Loss": expected_daily_pnl,
        "Total Frequency": total_frequency,
        "Long Frequency": long_frequency,
        "Short Frequency": short_frequency,
        "Sharpe Ratio": sharpe,
        "Annual Sharpe Ratio (252 trading days)": ann_sharpe,
        "Win Rate": win_rate,
        "Total Trades Performed": total_trades
    }
    
    return backtesting_results

def save_model(params: dict, backtesting: dict, confusion: dict):
    print("Model saving now...")
    model_save_path = os.path.join(saved_param_dir, "lr_model.joblib")
    dump(grid.best_estimator_, model_save_path)
    
    params = grid.best_params_
    params.update({"class_weight": class_weight})
    
    config = {
        "model_name": "Logistic Regression",
        "labels": 3,
        
        "params": params,
        
        "backtesting_result": backtesting,
        
        "confusion_matrix": confusion,
        
        "random_state": 67,
    }
    
    config_path = os.path.join(saved_param_dir, "json_files", "num_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
        
    print(f"Model saved successfully at {model_save_path}")
    print(f"Params saved successfully at {config_path}")
    
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(report)
print(confusion)
    
best_params = grid.best_params_
best_params.update({"class_weight": class_weight})

backtesting_results = backtesting(y_pred, X_test)

save = str(input("Save model? (y/n): "))

if save.lower().strip() == "y":
    save_model(best_params, backtesting_results, confusion.tolist())
else:
    print("Model is not saved. Previous saved params and models are untouched")