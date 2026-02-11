import torch
import torch.nn.functional as F

import os
import json

import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
    with torch.no_grad():
        for batch_input_ids, batch_attention_mask, batch_labels in dataloader:
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.to(device)
            
            output = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            
            if threshold_tuned:
                probabilities = F.softmax(output.logits, dim=1)
                
                for pred, true in zip(probabilities, batch_labels):
                    if torch.argmax(pred) == untuned_class:
                        label = untuned_class
                    else:
                        label = tune_class if pred[tune_class] > threshold else fallback_class
                
                    y_pred.append(label)
                    y_true.append(true.cpu().item())
            else:
                predictions = torch.argmax(output.logits, dim=1)
                
                y_pred.append(predictions.cpu())
                y_true.append(batch_labels.cpu())    
    
    y_pred = y_pred if threshold_tuned else torch.cat(y_pred).numpy()
    y_true = y_true if threshold_tuned else torch.cat(y_true).numpy()
    
    report = classification_report(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    
    return report, confusion
       
if __name__ == "__main__": 
    # print(predict_sentiment("Intel stock plunges as hopes for a 'clean' turnaround story meet reality"))
    
    import pandas as pd
    csv_dir = os.path.join(os.getcwd(), "csv_files")
    
    df = pd.read_csv(os.path.join(csv_dir, "raw", "enhanced_phrasebank.csv"), index_col=0)
    
    r, c = test_nlp_model(df, threshold_tuned=False)
    print(r)
    print(c)
    
    
    
    