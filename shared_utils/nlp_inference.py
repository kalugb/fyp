import torch
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib
matplotlib.use("Agg")

working_dir = os.getcwd()
saved_model_dir = os.path.join(working_dir, "saved_model_params")
finbert_utils_path = os.path.join(saved_model_dir, "finbert_utils")
token_utils_path = os.path.join(saved_model_dir, "tokenizer_utils")
json_path = os.path.join(saved_model_dir, "json_files", "nlp_config.json")

device = "cuda" if torch.cuda.is_available else "cpu"
with open(json_path, "r") as file:
    f = json.load(file)
    model_name = f["model_name"]
    tokenizer_params = f["tokenizer_params"]
    random_state = f["random_state"]
    threshold = f["hyperparameters_list"]["threshold_0"]
    batch_size = f["hyperparameters_list"]["batch_size"]
    
torch.cuda.manual_seed(random_state)
torch.manual_seed(random_state)
np.random.seed(random_state)

tokenizer = AutoTokenizer.from_pretrained(token_utils_path, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(finbert_utils_path, num_labels=3).to(device)

model.eval()

def save_graph(y, output_probs):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
    from sklearn.preprocessing import LabelBinarizer
    
    y = np.array(y)
    output_probs = np.array(output_probs)
    
    # PRC curve
    lb = LabelBinarizer()
    y_test_binarized = lb.fit_transform(y)
    classes = ["Negative", "Neutral", "Positive"]
    n_classes = len(classes)
    
    precision, recall, average_precision = dict(), dict(), dict()
    threshold = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], threshold[i] = precision_recall_curve(y_test_binarized[:, i], output_probs[:, i])
        average_precision[i] = average_precision_score(y_test_binarized[:, i], output_probs[:, i])
        
        plt.plot(recall[i], precision[i], label=f"{classes[i]} (AP = {average_precision[i]:.2f})")
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multi-clas PRC curve")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig("website/static/images/model_testing_nlp/prc_curve.jpg", dpi=300, bbox_inches="tight")
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
    plt.savefig("website/static/images/model_testing_nlp/roc_curve.jpg", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Threshold tuning graph for class 0 and 1 (tuned class)
    f1_negative = 2 * (precision[0][:-1] * recall[0][:-1]) / (precision[0][:-1] + recall[0][:-1])
    f1_neutral = 2 * (precision[1][:-1] * recall[1][:-1]) / (precision[1][:-1] + recall[1][:-1])
    plt.plot(threshold[0], precision[0][:-1], "-", color="red", label="Precision for Negative")
    plt.plot(threshold[0], recall[0][:-1], "-", color="orange", label="Recall for Negative")
    plt.plot(threshold[0], f1_negative, "-", color="yellow", label="F1 for Negative")
    plt.plot(threshold[1], precision[1][:-1], "-", color="green", label="Precision for Neutral")
    plt.plot(threshold[1], recall[1][:-1], "-", color="blue", label="Recall for Neutral")
    plt.plot(threshold[1], f1_neutral, "-", color="indigo", label="F1 for Neutral")
    plt.xlabel("Threshold")
    plt.ylabel("Scoring")
    plt.title("Scoring at various threshold for class Negative and Neutral")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("website/static/images/model_testing_nlp/threshold_tuning_class0_class1.jpg", dpi=300, bbox_inches="tight")
    plt.close()

def predict_sentiment(text):
    encoding = tokenizer(text, **tokenizer_params)
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = output.logits
        
        probabilities = F.softmax(logits, dim=1).cpu().numpy().squeeze(0)
        sentiment_label = torch.argmax(logits, dim=1).cpu().numpy()
        sentiment = ""
        
        probability_list = f"Negative: {probabilities[0].item():.4f}, Neutral: {probabilities[1].item():.4f}, Positive: {probabilities[2].item():.4f}"

        if sentiment_label == 0:
            sentiment = "Negative"
        elif sentiment_label == 1:
            sentiment = "Neutral"
        elif sentiment_label == 2:
            sentiment = "Positive"
        else:
            sentiment = "Undefined"
            print("Something went wrong. You shouldn't be here") 
        
        return sentiment, (sentiment_label - 1).item(), probability_list
    
def test_nlp_model(df, threshold_tuned=True, tune_class=0, fallback_class=1, untuned_class=2):
    from sklearn.metrics import confusion_matrix, classification_report
    from torch.utils.data import TensorDataset, DataLoader
    
    df = df.rename(columns={df.columns[0]: "text", df.columns[1]: "label"})
    
    X = df["text"].astype(str).tolist()
    y = df["label"].tolist()
    
    X_encoding = tokenizer(X, **tokenizer_params)
    X_input_ids = X_encoding["input_ids"]
    X_attention_mask = X_encoding["attention_mask"]
    y = torch.tensor(y, dtype=torch.long).to(device)
    
    dataset = TensorDataset(X_input_ids, X_attention_mask, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    y_pred = []
    y_true = []
    output_probs = []
    with torch.no_grad():
        for batch_input_ids, batch_attention_mask, batch_labels in dataloader:
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.to(device)
            
            output = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            probabilities = F.softmax(output.logits, dim=1)
            
            if threshold_tuned:
                for pred, true in zip(probabilities, batch_labels):
                    if torch.argmax(pred) == untuned_class:
                        label = untuned_class
                    else:
                        label = tune_class if pred[tune_class] > threshold else fallback_class
                
                    y_pred.append(label)
                    y_true.append(true.cpu().item())
                    output_probs.append(pred.detach().cpu().numpy())
            else:
                predictions = torch.argmax(output.logits, dim=1)
                
                y_pred.append(predictions.cpu())
                y_true.append(batch_labels.cpu())    
                output_probs.append(probabilities.detach().cpu().numpy())
    
    y_pred = y_pred if threshold_tuned else torch.cat(y_pred).numpy()
    y_true = y_true if threshold_tuned else torch.cat(y_true).numpy()
    raw_y = y.detach().cpu().numpy()
    output_probs = np.array(output_probs)
    
    confusion = confusion_matrix(y_true, y_pred)
    
    save_graph(raw_y, output_probs)
    
    return confusion
       
if __name__ == "__main__": 
    matplotlib.use("TkAgg")
    print(predict_sentiment("Intel stock plunges as hopes for a 'clean' turnaround story meet reality"))
    
    import pandas as pd
    csv_dir = os.path.join(os.getcwd(), "csv_files")
    
    df = pd.read_csv(os.path.join(csv_dir, "raw", "enhanced_phrasebank.csv"), index_col=0)
    
    c = test_nlp_model(df, threshold_tuned=True)
    print(c)
    print(type(c))    
    
    
    