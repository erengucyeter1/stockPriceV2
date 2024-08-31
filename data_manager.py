import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from datetime import date, datetime, timedelta

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm


class FileManager:
    def __init__(self, ticker):

        # set ticker name
        self.ticker = ticker

        # define folder paths
        self.data_path = "data/"
        self.ticker_path = f"{self.data_path}{self.ticker}/"
        self.news_path = f"{self.ticker_path}news/"
        self.raw_path = f"{self.ticker_path}raw/"
        self.processed_path = f"{self.ticker_path}processed/"
        self.fred_path = f"data/fred/"

        # define file paths
        self.raw_file = f"{self.raw_path}{self.ticker}.csv"
        self.news_file = f"{self.news_path}{self.ticker}.csv"
        self.processed_file = f"{self.processed_path}{self.ticker}.csv"
        self.prcessed_with_news_file = (
            f"{self.processed_path}{self.ticker}_with_news.csv"
        )
        self.processed_with_news_and_fred_file = (
            f"{self.processed_path}{self.ticker}_with_news_and_fred.csv"
        )
        self.news_average_score_file = (
            f"{self.news_path}{self.ticker}_average_scores.csv"
        )
        self.news_total_score_file = f"{self.news_path}{self.ticker}_total_scores.csv"
        self.news_score_file = f"{self.news_path}{self.ticker}_scores.csv"

        # create folders
        self.create_folders()

        # set start date
        if os.path.exists(self.raw_file):
            self.start_date = self.to_date(self.raw["Date"][0])
            self.end_date = self.to_date(self.raw["Date"].iloc[-1])
        else:
            self.start_date = None

    @property
    def raw(self):
        if os.path.exists(self.raw_file):
            return pd.read_csv(self.raw_file)
        else:
            raise FileNotFoundError(f"{self.raw_file} not found.")

    @raw.setter
    def raw(self, value):
        value.to_csv(self.raw_file, index=False)

    @property
    def news(self):
        if os.path.exists(self.news_file):
            return pd.read_csv(self.news_file)
        else:
            raise FileNotFoundError(f"{self.news_file} not found.")

    @property
    def news_score(self):
        if os.path.exists(self.news_score_file):
            return pd.read_csv(self.news_score_file)
        else:
            raise FileNotFoundError(f"{self.news_score_file} not found.")

    @property
    def news_average_score(self):
        if os.path.exists(self.news_average_score_file):
            return pd.read_csv(self.news_average_score_file)
        else:
            raise FileNotFoundError(f"{self.news_average_score_file} not found.")

    @property
    def news_total_score(self):
        if os.path.exists(self.news_total_score_file):
            return pd.read_csv(self.news_total_score_file)
        else:
            raise FileNotFoundError(f"{self.news_total_score_file} not found.")

    @news.setter
    def news(self, value):
        value.to_csv(self.news_file, index=False)

    @property
    def processed(self):
        if os.path.exists(self.processed_file):
            return pd.read_csv(self.processed_file)
        else:
            raise FileNotFoundError(f"{self.processed_file} not found.")

    @property
    def processed_with_news(self):
        if os.path.exists(self.prcessed_with_news_file):
            return pd.read_csv(self.prcessed_with_news_file)
        else:
            raise FileNotFoundError(f"{self.prcessed_with_news_file} not found.")

    @property
    def processed_with_news_and_fred(self):
        if os.path.exists(self.processed_with_news_and_fred_file):
            return pd.read_csv(self.processed_with_news_and_fred_file)
        else:
            raise FileNotFoundError(
                f"{self.processed_with_news_and_fred_file} not found."
            )

    def __get_df_info(self, path):

        if os.path.exists(path):
            df = pd.read_csv(path)
        else:
            return "File not found."

        start_date = df["Date"].iloc[0]
        end_date = df["Date"].iloc[-1]
        total_row = df.shape[0]
        return (
            f"start date: {start_date}\n end date: {end_date}\ntotal row: {total_row}\n"
        )

    def __str__(self) -> str:
        return f"""
Ticker: {self.ticker}
Raw data info: {self.__get_df_info(self.raw_file).strip()}
News data info: {self.__get_df_info(self.news_file).strip()}
Processed data info: {self.__get_df_info(self.processed_file).strip()}
    """

    def to_date(self, string: str) -> date:
        return datetime.strptime(string, "%Y-%m-%d").date()

    def today(self):
        return date.today()

    def create_folders(self):

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        if not os.path.exists(self.ticker_path):
            os.makedirs(self.ticker_path)
        if not os.path.exists(self.raw_path):
            os.makedirs(self.raw_path)
        if not os.path.exists(self.news_path):
            os.makedirs(self.news_path)
        if not os.path.exists(self.processed_path):
            os.makedirs(self.processed_path)

    # helper functions

    def file_already_exist_dialog(self, file_path):

        if os.path.exists(file_path):
            key = input("Data already exists. Do you want to overwrite it? (y/n)\n")
            while True:

                if key == "y":
                    return True
                elif key == "n":
                    return False
                else:
                    key = input("Invalid input. Please enter 'y' or 'n'.\n")


class DataManager(FileManager):
    def __init__(self, ticker):
        super().__init__(ticker)

    def update_data(self):

        if self.start_date is None:
            flag = True
            while flag:

                start_date = input(
                    "No data found. Please enter a start date:\nExample: '2020-01-01'"
                )
                if start_date == "e":
                    break
                try:
                    self.start_date = self.to_date(start_date)
                    self.download_data(self.start_date)
                    return self.raw
                except:
                    print("Invalid date format. Please use 'YYYY-MM-DD'.")
                    print("if you want to exit, type 'e'")

        else:
            today = self.today()

            if self.end_date < today:

                new_data = yf.download(self.ticker, start=self.end_date, end=today)
                new_data.reset_index(inplace=True)
                new_data.rename(columns={"index": "Date"}, inplace=True)
                new_data["Date"] = new_data["Date"].apply(lambda x: str(x.date()))

                self.raw = pd.concat([self.raw, new_data], ignore_index=True)
                self.start_date = self.to_date(self.raw["Date"][0])
                return self.raw

    def download_data(self, start_date):

        answer = self.file_already_exist_dialog(self.raw_file)

        if answer:
            data = yf.download(self.ticker, start=start_date, end=self.today())
            data.to_csv(self.raw_file)
            return data
        else:
            return self.raw


class NewsManager(FileManager):

    def __init__(self, ticker):
        super().__init__(ticker)

    def get_page_date(self, page):
        url = f"https://markets.businessinsider.com/news/{self.ticker.lower()}-stock?p={page}"
        response = requests.get(url)
        html = response.text
        soup = BeautifulSoup(html, "lxml")
        articles = soup.find_all("div", class_="latest-news__story")

        date_list = []

        for artickle in articles:

            date_time = artickle.find("time", class_="latest-news__date").get(
                "datetime"
            )

            date_time = date_time.split()[0]  # get only date. format: "MM, DD, YYYY"

            splited_date = date_time.split("/")

            curr_date = date(
                int(splited_date[2]), int(splited_date[0]), int(splited_date[1])
            )

            date_list.append(curr_date)

        return sorted(list(set(date_list)))

    def find_page_number(self, searching_date, tolerance=5, max_search=3000):

        searching_date = self.to_date(searching_date)

        page = 1
        last_page = 1
        step = int(((self.today() - searching_date).days / 10) + 0.5)
        while max_search > 0:
            max_search -= 1
            # input("Press Enter to continue...")
            page_dates = self.get_page_date(page)
            print(page_dates)
            input("page dates:")

            if searching_date in page_dates:
                print(f"page {page} is found")
                print(f"date: {page_dates[0]}")
                print(f"searching date: {searching_date}")
                return page

            if page_dates[-1] > searching_date:
                print("ikinci")
                last_page = page
                page = new_page
                continue

            new_page = int((page + last_page) / 2)

            new_date = self.get_page_date(new_page)

            if searching_date > new_date[0]:

                print("ilk")
                last_page = page
                page += step

            else:
                print("ucuncu")
                page = new_page

        return -1

    def get_news_data(self, page):

        column_list = ["Date", "Title", "Source", "Link"]
        all_rows = []

        while page > 0:
            url = f"https://markets.businessinsider.com/news/{self.ticker.lower()}-stock?p={page}"
            print(f"page {page} is downloading")
            response = requests.get(url)
            html = response.text
            soup = BeautifulSoup(html, "lxml")
            articles = soup.find_all("div", class_="latest-news__story")
            page -= 1  # page numberden geri gel !

            for artickle in articles[::-1]:

                date_time = artickle.find("time", class_="latest-news__date").get(
                    "datetime"
                )

                date_time = date_time.split()[
                    0
                ]  # get only date. format: "MM, DD, YYYY"

                splited_date = date_time.split("/")

                date_time = date(
                    int(splited_date[2]), int(splited_date[0]), int(splited_date[1])
                )
                title = artickle.find("a", class_="news-link").text

                source = artickle.find("span", class_="latest-news__source").text

                link = artickle.find("a", class_="news-link").get("href")

                all_rows.append([date_time, title, source, link])

        return pd.DataFrame(all_rows, columns=column_list)

    def download_news(self, start_date):
        page = self.find_page_number(start_date)
        df = self.download_get_news_data(page)
        df.to_csv(self.news_file, index=False)

    def update_news(self):

        last_date = self.news.tail(1)["Date"].values[0]

        start_page = self.find_page_number(last_date, tolerance=0)

        df = self.get_news_data(start_page)

        self.news = pd.concat([self.news, df], ignore_index=True)


class FredManager:
    def __init__(self, series_id_list=None):

        # get FRED API key
        self.fred_api_key = self.get_fred_api_key()

        if series_id_list is None:
            self.series_id_list = [
                "T10YIE",
                "GS10",
                "GDP",
                "CORESTICKM159SFRBATL",
                "UNRATE",
                "CSCICP03USM665S",
                "PPIACO",
            ]
        else:
            self.series_id_list = series_id_list

        self.fred_path = f"data/fred/"

        if not os.path.exists(self.fred_path):
            os.makedirs(self.fred_path)

    def __str__(self) -> str:
        return f"Serie ID List: {self.series_id_list}"

    def get_fred_api_key(self):

        if not os.path.exists(".env"):
            print("You must have a FRED API key to use this program.")
            api_key = input("FRED API Key: ")

            with open(".env", "w") as f:
                f.write(f"FRED_API_KEY={api_key}\n")
            print("API key has been saved to '.env' file.")

        else:
            load_dotenv()
            api_key = os.getenv("FRED_API_KEY")

        return api_key

    # cash flow verisi el ile alındı

    def __download_fred_data(self, start_date, series_id):

        print(f"Downloading {series_id} data...")

        start_date = "/".join(start_date.split("-")[::-1])
        fred = Fred(api_key=self.fred_api_key)
        data = fred.get_series(series_id, observation_start=start_date)
        df = pd.DataFrame(columns=["Date", "Value"])
        df["Date"] = data.index
        df["Value"] = data.values
        df.to_csv(f"{self.fred_path}{series_id}.csv", index=False)

    def get_fred_data(self, start_date, series_id=None):

        if series_id is None:

            for series_id in self.series_id_list:
                self.__download_fred_data(start_date, series_id)
        else:
            self.__download_fred_data(start_date, series_id)


class DataProcessing:
    def __init__(self, ticker):
        self.ticker = ticker
        self.fm = FileManager(self.ticker)

    def __str__(self) -> str:
        return f""""""

    @property
    def date_column(self):
        return self.fm.raw["Date"]

    def fill_missing_dates(self, path):

        files = os.listdir(path)

        for file in files:
            df = pd.read_csv(f"{path}{file}")

            new_df = pd.merge(self.date_column, df, on="Date", how="outer")
            if new_df.isnull().sum().sum() > 0:
                new_df = new_df.ffill()
                new_df = new_df.bfill()
            new_df.to_csv(f"{path}{file}", index=False)


class NewsProcessing:

    def __init__(self, ticker):
        self.ticker = ticker
        self.fm = FileManager(self.ticker)
        self.score_path = f"{self.fm.news_path}{self.ticker}_scores.csv"

        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert"
        )

    @property
    def last_predicted_date(self):
        if os.path.exists(self.score_path):
            df = pd.read_csv(self.score_path)
            return self.fm.to_date(df["Date"].iloc[0])
        else:
            return None

    def __predict(self, text):

        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=16384, truncation=True
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits

        rates = logits.softmax(dim=1).tolist()

        df = pd.DataFrame(rates, columns=self.model.config.id2label.values())
        return df

    def predict_sentiment_score(self):

        df_columns = ["Date"]
        df_columns += self.model.config.id2label.values()

        if os.path.exists(self.score_path):
            sentiment_df = pd.read_csv(self.score_path)
        else:
            sentiment_df = pd.DataFrame(columns=self.model.config.id2label.values())

        if self.last_predicted_date is None:
            news = self.fm.news
        else:
            index = self.fm.news[
                self.fm.news["Date"].apply(self.fm.to_date) == self.last_predicted_date
            ].index.max()
            news = self.fm.news.iloc[index + 1 :]
            date_column = news.pop("Date")
            date_column.reset_index(drop=True, inplace=True)

        results = []

        for i, text in tqdm(enumerate(news["Title"])):

            sentiment = self.__predict(text)

            results.append(
                {
                    "Date": date_column.iloc[i],
                    "positive": sentiment["positive"].iloc[0],
                    "negative": sentiment["negative"].iloc[0],
                    "neutral": sentiment["neutral"].iloc[0],
                }
            )

        new_df = pd.DataFrame(
            results,
            columns=df_columns,
        )

        new_df = new_df[::-1]
        sentiment_df = pd.concat([new_df, sentiment_df], ignore_index=True)
        sentiment_df.to_csv(self.score_path, index=False)


if __name__ == "__main__":

    file_manager = FileManager("MSFT")
    fm = FredManager()
    dp = DataProcessing("MSFT")

    dp.fill_missing_dates(file_manager.fred_path)
