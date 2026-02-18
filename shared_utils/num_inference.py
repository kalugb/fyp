import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib
matplotlib.use("Agg")

from shared_utils.load_utils import load_num_model

model = load_num_model()

def save_graph(X, y, output_probs):
    from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
    from sklearn.model_selection import learning_curve
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.linear_model import LogisticRegression
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    import json
    import numpy as np
    
    # PRC curve
    lb = LabelBinarizer()
    y_test_binarized = lb.fit_transform(y)
    classes = ["Short", "Hold", "Long"]
    n_classes = len(classes)
    
    precision, recall, average_precision = dict(), dict(), dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], output_probs[:, i])
        average_precision[i] = average_precision_score(y_test_binarized[:, i], output_probs[:, i])
        
        plt.plot(recall[i], precision[i], label=f"{classes[i]} (AP = {average_precision[i]:.2f})")
        
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multi-class PRC curve")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig("website/static/images/model_testing_num/prc_curve.jpg", dpi=300, bbox_inches="tight")
    plt.close()
    
    # ROC curve
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], output_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.plot(fpr[i], tpr[i], label=f"{classes[i]} (AUC = {roc_auc[i]:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
       
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve for all classes")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig("website/static/images/model_testing_num/roc_curve.jpg", dpi=300, bbox_inches="tight")
    plt.close()
    
    # learning curve
    with open("saved_model_params/json_files/num_config.json", "r") as file:
        data = json.load(file)
        
        parameters = data["params"]
        max_iter = parameters["lr__max_iter"]
        C = parameters["lr__C"]
        class_weight = {float(k): v for k, v in parameters["class_weight"].items()}
        
    using_model = LogisticRegression(max_iter=max_iter, C=C, class_weight=class_weight,
                               verbose=0, random_state=67, n_jobs=-1)
        
    train_size, train_scores, val_scores = learning_curve(
        using_model, X, y,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=3,
        scoring="precision_macro",
        shuffle=True,
        random_state=67,
        n_jobs=-1
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
     
    plt.plot(train_size, train_mean, "o-", color="blue", label="Training Score")
    plt.fill_between(train_size, train_mean + train_std, train_mean - train_std, alpha=0.15, color="blue")
    
    plt.plot(train_size, val_mean, "o-", color="green", label="Validation Score")
    plt.fill_between(train_size, val_mean + val_std, val_mean - val_std, alpha=0.15, color="green")
    
    plt.xlabel("Training Set Size")
    plt.ylabel("Score")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig("website/static/images/model_testing_num/learning_curve.jpg", dpi=300, bbox_inches="tight")
    plt.close()

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
    from sklearn.metrics import confusion_matrix
    
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
    output_probs = model.predict_proba(X)
    
    confusion = confusion_matrix(y, output_list)
    
    save_graph(X, y, output_probs)
    
    return confusion
    
# placeholder, just for testing model
if __name__ == "__main__":  
    # print(predict_position(255.41000366210938, 0.029280115023907816, 261.2585681733631, 0.9776138843899168))
    import pandas as pd
    df_dir = os.path.join(os.getcwd(), "csv_files", "raw", "historical_stock_data.csv")
    df = pd.read_csv(df_dir, index_col=0)

    c = test_num_model(df)
    
    print(c)
    print(type(c))


