import pandas as pd


data = pd.read_csv("sentiment_processes/sentiment_analysed_result_with_content.csv")


print(data.head())

summed_data = data.groupby("Date").sum().reset_index()


summed_data.to_csv(
    "sentiment_processes/sentiment_analysis_result_summed_with_content.csv", index=False
)
