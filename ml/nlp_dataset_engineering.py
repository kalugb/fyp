import pandas as pd
import os
from collections import Counter

working_dir = os.getcwd()
csv_dir = os.path.join(working_dir, "csv_files")
phrase_fiqa_path = os.path.join(csv_dir, "raw", "phrasebank_fiqa.csv")
enhanced_phrase_path = os.path.join(csv_dir, "raw", "enhanced_phrasebank.csv")

df_ori = pd.read_csv(phrase_fiqa_path, index_col=0)
df_enh = pd.read_csv(enhanced_phrase_path, index_col=0)

mask = (df_enh["label"] == 0) | (df_enh["label"] == 2)

df_enh = df_enh.loc[mask, :].reset_index(drop=True)

df = pd.concat([df_ori, df_enh], ignore_index=True)

df["text"] = df["text"].str.strip()

df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

print(df)
print(Counter(df["label"]))

final_dataset_path = os.path.join(csv_dir, "cleaned", "merged_phrasebank.csv")
df.to_csv(final_dataset_path)


# dataset info 
# dataset total rows: 5926 rows
# Class 0: 1196
# Class 1: 2878
# Class 2: 3215

