import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("csv_files/inference/historical_data.csv", index_col=0)

plt.figure(figsize=(16, 12))
plt.plot(df["Adj Close"][5000:], color="blue", label="Adjusted Closing Price")
plt.show()