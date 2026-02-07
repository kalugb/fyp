import time

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
    
torch.cuda.manual_seed(random_state)
torch.manual_seed(random_state)
np.random.seed(random_state)

tokenizer = AutoTokenizer.from_pretrained(token_utils_path, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(finbert_utils_path, num_labels=3).to(device)

model.eval()

def predict():
    text = "Intel stock plunges as hopes for a 'clean' turnaround story meet reality"
    
    encoding = tokenizer(text, **tokenizer_params)
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = output.logits
        
        probabilities = F.softmax(logits, dim=1).cpu().numpy().squeeze(0)
        prediction = torch.argmax(logits, dim=1).cpu().numpy()
        sentiment = ""
        
        probability_list = f"Negative: {probabilities[0].item():.4f}, Neutral: {probabilities[1].item():.4f}, Positive: {probabilities[2].item():.4f}"

        if prediction == 0:
            sentiment = "Negative"
        elif prediction == 1:
            sentiment = "Neutral"
        elif prediction == 2:
            sentiment = "Positive"
        else:
            print("Something went wrong. You shouldn't be here") 
            
        print(f"Sentiment: {sentiment}")
        print("Sentiment probability list:")
        print(probability_list)
       
if __name__ == "__main__": 
    predict()