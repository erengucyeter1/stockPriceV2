import pandas as pd
import os
import seaborn as sns


df = pd.read_csv("data/MSFT/processed_data/MSFT.csv")


# Calculate the correlation matrix
corr_matrix = df.drop("Date", axis=1).corr()

# Plot the correlation matrix
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
