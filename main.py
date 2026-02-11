import pandas as pd
import os
csv_dir = os.path.join(os.getcwd(), "csv_files")

df = pd.read_csv(os.path.join(csv_dir, "cleaned", "merged_phrasebank.csv"), index_col=0)

print(df.columns)

df = df.rename(columns={df.columns[1]: "label"})

print(df, df.columns)