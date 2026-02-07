import pandas as pd
import string
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter
import os

working_dir = os.getcwd()
current_dir = os.path.dirname(__file__)
csv_dir = os.path.join(working_dir, "csv_files")

df_path = os.path.join(csv_dir, "merged_phrasebank.csv")
df = pd.read_csv(df_path, index_col=0)

def show_wordcloud(df, save_as_image=False):
    remove_punctuation = str.maketrans("", "", string.punctuation)
    stopword = set(STOPWORDS)

    df["text"] = df["text"].str.translate(remove_punctuation)
    text = " ".join(df["text"].astype(str).tolist()).lower()
    text = " ".join(word for word in text.split() if word not in stopword and len(word) >= 3)

    wordcloud = WordCloud(width=1600, height=800, background_color="white", random_state=67).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  
    plt.title("Combined Phrasebank Word Cloud")
    plt.show()
    
    if save_as_image:
        path = os.path.join(current_dir, "phrasebank_wordcloud.png")
        wordcloud.to_file(path)
        
        print(f"Wordcloud saved successfully in {path}")
    
def show_label_count(df, save_as_image=False):
    labels = df["label"]
    labels = labels.map({0: "Negative", 1: "Neutral", 2: "Positive"})
    
    labels = Counter(labels)
    labels = dict(sorted(labels.items()))
    
    print(labels)
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels.keys(), labels.values())
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(f"Histogram of label counts: {labels}")
    
    if save_as_image:
        path = os.path.join(current_dir, "label_counts.png")
        plt.savefig(path, dpi=72, bbox_inches="tight")
        print(f"Histogram saved successfully in {path}")
    
    plt.show()
    
show_wordcloud(df, True)
show_label_count(df, True)