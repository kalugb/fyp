import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared_utils.load_utils import load_num_model

model = load_num_model()

def predict_position(adj_close, returns, ma, ratio):
    data = [[adj_close, returns, ma, ratio]] 
    columns = ["Adj Close", "returns", "ma", "ratio"]

    dataframe = pd.DataFrame(
        data,
        columns=columns
    )

    probabilities = model.predict_proba(dataframe).squeeze(0)
    classes = model.classes_
    position_label = classes[probabilities.argmax()]
    position = None

    if position_label == -1:
        position = "Short"
    elif position_label == 0:
        position = "Hold"
    elif position_label == 1:
        position = "Long"
    else:
        print(f"Prediction not defined. Something went wrong")
    
    return position, int(position_label), probabilities.tolist()

def test_num_model(df, perform_mean_reversion=True):
    from sklearn.metrics import confusion_matrix, classification_report
    
    if perform_mean_reversion:
        from website.backend_functions import mean_reversion
        print("Performing mean reversion...")
        df = df.rename(columns={df.columns[0]: "Open", df.columns[1]: "High", 
                        df.columns[2]: "Low", df.columns[3]: "Close", 
                        df.columns[4]: "Adj Close", df.columns[5]: "Volume"})
        df = mean_reversion(df)
        df = df.dropna()
    else:
        print("Not performing mean reversion...")
        df = df.rename(columns={df.columns[0]: "Adj Close", df.columns[1]: "returns", 
                        df.columns[2]: "ma", df.columns[3]: "ratio", 
                        df.columns[4]: "position"})
    
    X = df.drop(columns="position")
    y = df["position"]
    output_list = model.predict(X)
    
    report = classification_report(y, output_list)
    confusion = confusion_matrix(y, output_list)
    
    return report, confusion
    
# placeholder, just for testing model
if __name__ == "__main__":  
    # print(predict_position(255.41000366210938, 0.029280115023907816, 261.2585681733631, 0.9776138843899168))
    import pandas as pd
    df_dir = os.path.join(os.getcwd(), "csv_files", "raw", "historical_stock_data.csv")
    df = pd.read_csv(df_dir, index_col=0)
    
    r, c = test_num_model(df)
    
    print(r, c)


