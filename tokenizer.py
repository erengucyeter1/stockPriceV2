## tüm file lerden veri çek işleme hazır hale getir
from data_manager import FileManager
import pandas as pd
import os


class MyTokenizer:

    def __init__(self, ticker):
        self.ticker = ticker
        self.fm = FileManager(ticker)

    def get_unique_news_average_score(self):

        news_score = self.fm.news_score

        average_news_score = news_score.groupby("Date").mean().reset_index()

        average_news_score.to_csv(
            f"{self.fm.news_path}{self.ticker}_average_scores.csv", index=False
        )

    def get_unique_news_total_score(self):

        news_score = self.fm.news_score

        average_news_score = news_score.groupby("Date").sum().reset_index()

        average_news_score.to_csv(
            f"{self.fm.news_path}{self.ticker}_total_scores.csv", index=False
        )

    def merge_data(self):
        ticker_data = self.fm.raw
        news_data = self.fm.news_total_score  # or news_average_score

        merged_data = pd.merge(ticker_data, news_data, on="Date", how="outer")

        index_list = []
        for index in range(len(merged_data) - 1, 0, -1):

            row = merged_data.iloc[index]
            if pd.isnull(row["Close"]):

                merged_data.iloc[index - 1, -3:] += row.iloc[-3:]
                index_list.append(index)

        merged_data.drop(index_list, inplace=True)
        merged_data.ffill(inplace=True)
        merged_data.dropna(inplace=True)

        merged_data.to_csv(
            f"{self.fm.processed_path}{self.ticker}_with_news.csv", index=False
        )

    def merge_fred_data(self):

        ticker_data = self.fm.processed_with_news
        date_column = ticker_data["Date"]

        fred_list = os.listdir(self.fm.fred_path)
        fred_column_names = ["Date"] + [file.split(".")[0] for file in fred_list]

        only_fred_data = pd.DataFrame(columns=fred_column_names)
        only_fred_data["Date"] = date_column

        for file in fred_list:

            series_id = file.split(".")[0]
            fred_data = pd.read_csv(self.fm.fred_path + file)

            only_fred_data[series_id] = fred_data["Value"]

        ticker_data = pd.merge(ticker_data, only_fred_data, on="Date", how="inner")

        ticker_data.to_csv(
            f"{self.fm.processed_path}{self.ticker}_with_news_and_fred.csv", index=False
        )


if __name__ == "__main__":

    tokenizer = MyTokenizer("MSFT")

    tokenizer.merge_data()
