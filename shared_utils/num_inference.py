import pandas as pd

from shared_utils.load_utils import load_num_model

model = load_num_model()

# will continue after having inference plan
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
    
# placeholder, just for testing model
if __name__ == "__main__":  
    print(predict_position(255.41000366210938, 0.029280115023907816, 261.2585681733631, 0.9776138843899168))

