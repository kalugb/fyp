import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

from collections import Counter

import json

import os

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))

saved_model_path = os.path.join(os.getcwd(), "saved_model_params")
json_path = os.path.join(saved_model_path, "json_files")
csv_path = os.path.join(os.getcwd(), "csv_files")

num_json_path = os.path.join(json_path, "num_config.json")
nlp_json_path = os.path.join(json_path, "nlp_config.json")

historical_stock_dataset_path = os.path.join(csv_path, "cleaned_historical_stock_data.csv")
nlp_dataset_path = os.path.join(csv_path, "merged_phrasebank.csv")

def defining_model(model):
    pass

def loss_graph(loss: list[float]):
    plt.plot(loss, marker="o", label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss among {len(loss)} epoch(s)")
    
    plt.legend()
    plt.show()
    
def mean_reversion_line(ratio: pd.Series):
    percentiles = [20, 50, 80]
    
    percentiles = np.percentile(ratio.dropna(), percentiles)
    
    ratio.dropna().plot(legend=True, figsize=(10, 10))
    
    for p, color in zip(percentiles, ["gray", "lightgray", "black"]):
        plt.axhline(p, c=color, ls="--", label=f"Threshold at {p:.4f}")
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def n_common_word_bar(n: int, texts: pd.Series):
    import string
    from sklearn.feature_extraction import text
    
    def remove_stopwords_punctuation(sentence):
        sentence = sentence.translate(str.maketrans("", "", string.punctuation))
        sentence = sentence.lower()
        
        word_list = sentence.split()
        cleaned_sentence = " ".join([w for w in word_list if w not in text.ENGLISH_STOP_WORDS and len(w) >= 3])
        
        return cleaned_sentence
    
    texts = texts.dropna().apply(remove_stopwords_punctuation)
    
    top_n_words = Counter(" ".join(texts).split()).most_common(n)
    
    labels, values = zip(*top_n_words)
    indexes = np.arange(len(labels))
    
    plt.bar(indexes, values, width=0.5, color="skyblue")
    
    plt.xlabel("Words", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Top {n} most frequent words", fontsize=14)
    
    plt.xticks(indexes, labels, rotation=45, ha="right")
    
    plt.show()

df_num = pd.read_csv(historical_stock_dataset_path, index_col=0)
with open(num_json_path, "r") as file:
    pass

df_nlp = pd.read_csv(nlp_dataset_path, index_col=0)
with open(nlp_json_path, "r") as file:
    f = json.load(file)
    
    nlp_model_name = f["model_name"]
    all_loss = f["all_loss"]    
    
loss_graph(all_loss)
mean_reversion_line(df_num["ratio"])
n_common_word_bar(15, df_nlp["text"])
    
    