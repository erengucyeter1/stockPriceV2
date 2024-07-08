import yfinance as yf
import os
import pandas as pd
from feature_extraction import create_features


class GetData:
    def __init__(self, ticker, start=None, end=None):
        self.ticker = ticker
        if start is None:
            start = "2022-01-01"
        else:
            self.start = start

        if end is None:
            end = pd.Timestamp.today()
        else:
            self.end = end
        self.data_path = "data/"
        self.ticker_path = self.data_path + self.ticker + "/"
        self.raw_data_path = self.ticker_path + "raw_data/"
        self.processed_data_path = self.ticker_path + "processed_data/"
        self.split_data_path = self.ticker_path + "split_data/"

        self.__createFolder()

    # folder functions

    def __createFolder(self):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        if not os.path.exists(self.ticker_path):
            os.makedirs(self.ticker_path)
        if not os.path.exists(self.raw_data_path):
            os.makedirs(self.raw_data_path)
        if not os.path.exists(self.processed_data_path):
            os.makedirs(self.processed_data_path)
        if not os.path.exists(self.split_data_path):
            os.makedirs(self.split_data_path)

    # get data

    def get_raw_data(self):
        if os.path.exists(self.raw_data_path + self.ticker + ".csv"):
            data = pd.read_csv(self.raw_data_path + self.ticker + ".csv")

        else:
            data = yf.download(self.ticker, start=self.start, end=self.end)
            data.to_csv(self.raw_data_path + self.ticker + ".csv")

            data.reset_index(inplace=True)
            data.rename(columns={"index": "Date"}, inplace=True)
        return data

    def get_processed_data(self):
        if os.path.exists(self.processed_data_path + self.ticker + ".csv"):
            data = pd.read_csv(self.processed_data_path + self.ticker + ".csv")
        else:
            data = self.get_raw_data()
            data = create_features(data)
            data.to_csv(self.processed_data_path + self.ticker + ".csv")
        return data
