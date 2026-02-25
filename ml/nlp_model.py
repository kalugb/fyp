import os
import json

import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(67)
torch.cuda.manual_seed(67)
np.random.seed(67)

# get required directories
working_dir = os.getcwd()
param_dir = os.path.join(working_dir, "saved_model_params") # for saving model / params
pt_dir = os.path.join(param_dir, "pt_files") # Used for saving .pt files
json_dir = os.path.join(param_dir, "json_files") # Used for saving .json files
temp_model_path = os.path.join(param_dir, "temp_model.pt") # Used for early stopping mechanism and loading for test

df_path = os.path.join(working_dir, "csv_files", "cleaned", "merged_phrasebank.csv")
df = pd.read_csv(df_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

df = df.drop_duplicates(subset="text")

X = df["text"].astype(str).tolist()
y = df["label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=67, stratify=y_train)

print("Defining model...")
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
).to(device)

tokenizer_params = {
    "padding": True,
    "truncation": True,
    "max_length": 128,
    "return_tensors": "pt"
}

# Focal Loss is a specliazed loss function that focuses on example that models gets wrong rather than ones that can be predicted correctly
# Ensuring predictions on hard examples (minority) can be improved over time, rather than become overly confident with easy ones
# Uses down weighting, reduce the influence of easy examples on the loss functions, giving more attentions to the hard examples
# source: https://towardsdatascience.com/focal-loss-a-better-alternative-for-cross-entropy-1d073d92d075/

# Focal loss (FL) = ((1 - pt) ** gamma) * CE, where Cross Entropy (CE) = -ln(pt), pt is true class probability, where pt = exp(-CE)
# Full formula = - ((1 - pt) ** gamma) * ln(pt)
# If gamma = 0, FL = - ln(pt) = CE

# code source
# Original by Prasad G, (2020) in github (https://github.com/gokulprasadthekkel/pytorch-multi-class-focal-loss/blob/master/focal_loss.py)
# Modified by me in 19 Jan 2026
# Fixed: F_cross_entropy(..., reduction="none", ...), changed reduction=self.reduction -> reduction="none"
# Reason: reduction=self.reduction returns scalar, which breaks focal weighting, reduction="none" return each loss for each elements
class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor = None, gamma: float = 2, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight # act as alpha
        self.reduction = reduction
        
    def forward(self, input, target):
        # calculate CE and FL
        ce_loss = F.cross_entropy(input, target, reduction="none", weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction.lower().strip() == "mean":
            return focal_loss.mean()
        elif self.reduction.lower().strip() == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

def create_dataloader(X, y, tokenizer=tokenizer, tokenizer_params=tokenizer_params, batch_size=32, shuffle=True):
    encoding = tokenizer(X, **tokenizer_params)
    
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    y_data = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(input_ids, attention_mask, y_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

def early_stopping(val_dataloader, patience, best_val_f1):
    model.eval()
    val_pred = []
    val_true = []
    stop_signal = False
    
    # early stopping
    with torch.no_grad():
        for batch_input_ids, batch_attention_mask, batch_labels in val_dataloader:
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.to(device)
            
            val_output = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            
            val_label = torch.argmax(val_output.logits, dim=1)
            
            val_true.append(batch_labels.cpu())
            val_pred.append(val_label.cpu())
            
    val_true = torch.cat(val_true).numpy()
    val_pred = torch.cat(val_pred).numpy()

    val_f1 = f1_score(val_true, val_pred, average="macro")
    print(f"Current val_f1: {val_f1:.4f}")
    
    # reduce patience count if no improvement
    if val_f1 <= best_val_f1:
        patience -= 1
        print("Patience reduced")
        
        # Stop training if no improvement for 1 continuous epochs
        if patience == 0:
            stop_signal = True
    else:
        # update results, reset patience count, save current model with best results
        best_val_f1 = val_f1
        print("best_val_f1 updated")
        patience = 1
        torch.save(model.state_dict(), temp_model_path)
        
    return stop_signal, patience, best_val_f1

# training
def training(dataloader, criterion, optimizer, early_stop=True, val_dataloader=None, all_patience=None, return_loss=False):
    patience = all_patience
    all_loss = [] # save all loss per epoch (for graphing)
    best_val_f1 = 0 # for early stopping
    
    for ep in range(epochs):
        model.train()
        print(f"At epoch {ep + 1}")
        ep_loss = 0
        
        for batch_input_ids, batch_attention_mask, batch_labels in dataloader:
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.to(device)
            
            output = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            
            optimizer.zero_grad()
            
            loss = criterion(output.logits, batch_labels)
            ep_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        print(f"Avg loss at epoch {ep + 1}: {ep_loss / len(train_loader)}")
        all_loss.append(ep_loss / len(train_loader))
        
        if early_stop:
            stop_signal, patience, best_val_f1 = early_stopping(val_dataloader=val_dataloader, patience=patience, best_val_f1=best_val_f1)
            
            if stop_signal:
                print("Early stopping initiated...")
                return all_loss if return_loss else None
        else:
            torch.save(model.state_dict(), temp_model_path)
            
    return all_loss if return_loss else None

# threshold tuning for class 0 / 1
def threshold_tuning(val_dataloader, threshold_list=np.linspace(0.25, 0.4, 20), 
                    tune_class=0, fallback_class=1, untuned_class=2):
    threshold_tuned = True
    
    with torch.no_grad():
        best_f1 = 0
        best_threshold = 0
        
        for threshold in threshold_list:
            thres_preds = []
            thres_trues = []
            
            # threshold tuning
            for batch_input_ids, batch_attention_mask, batch_labels in val_dataloader:   
                batch_input_ids = batch_input_ids.to(device)
                batch_attention_mask = batch_attention_mask.to(device)
                batch_labels = batch_labels.to(device)         
                
                thres_output = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                thres_probs = F.softmax(thres_output.logits, dim=1)
                
                # apply threshold at the tuning class (class 0 against class 1), while leaving untuned class as it (class 2)
                for p, true_label in zip(thres_probs, batch_labels):
                    if torch.argmax(p) == untuned_class:
                        label = untuned_class
                    else:
                        label = tune_class if p[tune_class] > threshold else fallback_class
                        
                    thres_preds.append(label)
                    thres_trues.append(true_label.cpu().item())
                
            # calculate f1 score based on tuning class class 0 
            thres_f1 = f1_score(thres_trues, thres_preds, average="macro", labels=[tune_class])
            
            print(f"Current F1 with threshold {threshold}: {thres_f1}")
            
            # update best threshold based on f1 score calculated
            if thres_f1 > best_f1:
                best_f1 = thres_f1
                best_threshold = threshold
                
                print(f"Updated. Current best f1 with threshold {best_threshold}: {best_f1}")
                
    return best_threshold, threshold_tuned

def testing(test_dataloader, threshold_tuned=True, set_threshold=0.33, tune_class=0, fallback_class=1, untuned_class=2):
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_input_ids, batch_attention_mask, batch_labels in test_dataloader:
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.to(device)
            
            preds = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            
            if threshold_tuned:
                probabilities = F.softmax(preds.logits, dim=1)
                
                # apply threshold on class 0 / 1
                for pred, true_labels in zip(probabilities, batch_labels):
                    if torch.argmax(pred) == untuned_class:
                        label = untuned_class
                    else:
                        label = tune_class if pred[tune_class] > set_threshold else fallback_class
                
                    y_pred.append(label)
                    y_true.append(true_labels.cpu().item())
            else:
                predictions = torch.argmax(preds.logits, dim=1)
                
                y_pred.append(predictions.cpu())
                y_true.append(batch_labels.cpu())
                
    # only concatenate if using default threshold, as default append by batch, unlike threshold_tuned where it is append by rows
    y_pred = y_pred if threshold_tuned else torch.cat(y_pred).numpy()
    y_true = y_true if threshold_tuned else torch.cat(y_true).numpy()
    
    report = classification_report(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    
    return report, confusion

def save_model(gamma: float, reduction_type: str, weight: torch.Tensor | np.ndarray | list,
               train_dataset: torch.utils.data.Dataset, test_dataset: torch.utils.data.Dataset, 
               val_dataset: torch.utils.data.Dataset, tokenizer_params: dict, batch_size: int, 
               lr: float, weight_decay: float, epochs: int, 
               threshold: float, patience: int, all_loss: list[float], confusion: list):
    
    # save the model and with their huggingface utils (faster inference)
    model_pt_path = os.path.join(pt_dir, "best_finbert.pt")
    model_utils_path = os.path.join(param_dir, "finbert_utils")
    torch.save(model.state_dict(), model_pt_path)
    model.save_pretrained(model_utils_path)
    
    # save tokenizer utils (faster inference)
    token_utils_path = os.path.join(param_dir, "tokenizer_utils")
    tokenizer.save_pretrained(token_utils_path)
    
    # save train, val, test dataset into .pt file (faster inference, well...)
    train_dataset_path = os.path.join(pt_dir, "train_dataset.pt")
    test_dataset_path = os.path.join(pt_dir, "test_dataset.pt")
    val_dataset_path = os.path.join(pt_dir, "val_dataset.pt")
    torch.save(train_dataset, train_dataset_path)
    torch.save(val_dataset, val_dataset_path)
    torch.save(test_dataset, test_dataset_path)
    
    # convert weight to list
    if isinstance(weight, torch.Tensor):
        weight = weight.cpu().tolist()
    elif isinstance(weight, np.ndarray):
        weight = weight.tolist()
    else:
        weight = weight
    
    # parameters list
    config = {
        "model_name": "ProsusAI/finbert",
        "num_labels": 3,
        
        "loss_list": {
            "type": "FocalLoss",
            "gamma": gamma,
            "reduction_type": reduction_type,
            "weights": weight
        },
        
        "tokenizer_params": tokenizer_params,
        
        "hyperparameters_list": {
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "threshold_0": threshold,
            "patience": patience
        },
        
        "all_loss": all_loss,
        
        "confusion_matrix": confusion,
        
        "random_state": 67
    }
    
    json_path = os.path.join(json_dir, "nlp_config.json")
    with open(json_path, "w") as f:
        json.dump(config, f, indent=4)
        
    print(f"Model utils are saved at path: {model_utils_path}") 
    print(f"Model pth files are saved at path: {model_pt_path}")   
    print(f"Tokenizer utils are saved at path: {token_utils_path}")
    print(f"Train dataset is saved at path: {train_dataset_path}")
    print(f"Test dataset is saved at path: {test_dataset_path}")
    print(f"Val dataset is saved at path: {val_dataset_path}")
     
    print("Model saved successfully")
# end save_model()

# Hyperparameters list
batch_size = 32
lr = 2e-5
weight_decay = 0.001
epochs = 3

# FocalLoss parameters
gamma = 1.5
reduction_type = "mean"
weight = torch.tensor([2.0, 1.0, 1.0], dtype=torch.float32).to(device)

# loss and optimizer
criterion = FocalLoss(weight=weight, gamma=gamma, reduction=reduction_type)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# defining dataloader
train_loader = create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
val_loader = create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

print("model training now...")

# early stopping variables, used when early_stop=True
total_patience = 1
best_val_f1 = 0

# train, return all_loss for graphing later
all_loss = training(train_loader, criterion, optimizer, 
                    early_stop=True, val_dataloader=val_loader, 
                    all_patience=total_patience, return_loss=True)
    
# load model with best val results
model.load_state_dict(torch.load(temp_model_path))
model.eval()

# required threshold tuning variables
threshold_tuned = False
tune_class = 0
fallback_class = 1
untuned_class = 2
best_threshold, threshold_tuned = threshold_tuning(val_dataloader=val_loader, tune_class=tune_class, 
                                                   fallback_class=fallback_class, untuned_class=untuned_class)
        
# model testing 
print(f"Using threshold {best_threshold}")
print("model evaluation")
model.eval()

# create testloader
test_loader = create_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

# get metrics 
report, confusion = testing(test_dataloader=test_loader, threshold_tuned=threshold_tuned, set_threshold=best_threshold,
                         tune_class=tune_class, fallback_class=fallback_class, untuned_class=untuned_class)
report_train, confusion_train = testing(test_dataloader=train_loader, threshold_tuned=threshold_tuned, set_threshold=best_threshold,
                                     tune_class=tune_class, fallback_class=fallback_class, untuned_class=untuned_class)

print(report)
print(confusion)

print(report_train)
print(confusion_train)

save = str(input("Save the model? (y/n): "))

if save.lower().strip() == "y":
    print("Model saving in progress...")
    # Retrieve the dataset from DataLoader, making it more flexible as I can do .tensor, or rewrap in dataloader 
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset
    val_dataset = val_loader.dataset
    
    save_model(gamma, reduction_type, 
               weight, train_dataset, test_dataset,
               val_dataset, tokenizer_params, batch_size, 
               lr, weight_decay, epochs, 
               best_threshold, total_patience, all_loss, confusion.tolist())
else:
    print("Model is not saved. Previous saved params and models are untouched")
