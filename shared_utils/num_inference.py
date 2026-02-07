import pandas as pd

from load_utils import load_num_model

model = load_num_model()

# will continue after having inference plan
def predict(adj_close, returns, ma, ratio):
    data = [[adj_close, returns, ma, ratio]] # example data
    columns = ["Adj Close", "returns", "ma", "ratio"]

    dataframe = pd.DataFrame(
        data,
        columns=columns
    )

    probabilities = model.predict_proba(dataframe).squeeze(0)
    classes = model.classes_
    prediction = classes[probabilities.argmax()]
    position = None

    if prediction == -1:
        position = "Short"
    elif prediction == 0:
        position = "Hold"
    elif prediction == 1:
        position = "Long"
    else:
        print(f"Prediction not defined. Something went wrong")
    
    print(f"Prediction: {position}")
    print(f"Probabilities: {probabilities.tolist()}")
    
# placeholder, just for testing model
if __name__ == "__main__":  
    predict(255.41000366210938, 0.029280115023907816, 261.2585681733631, 0.9776138843899168)

