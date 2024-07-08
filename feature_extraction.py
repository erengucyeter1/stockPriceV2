import numpy as np
import pandas as pd


def calculate_ema(data, day):  # Exponential Moving Average
    data["EMA"] = data["Close"].ewm(span=day, adjust=False).mean()


def calculate_bollinger_bands(data, day):  # Bollinger Bands
    data["SMA"] = data["Close"].rolling(window=day).mean()
    data["STD"] = data["Close"].rolling(window=day).std()
    data["Upper"] = data["SMA"] + (data["STD"] * 2)
    data["Lower"] = data["SMA"] - (data["STD"] * 2)


def calculate_rate_of_change(data, day):  # Rate of Change
    data["ROC"] = data["Close"].pct_change(periods=day)


def calculate_commodity_channel_index(data, day):  # Commodity Channel Index
    TP = (data["High"] + data["Low"] + data["Close"]) / 3
    data["SMA_TP"] = TP.rolling(window=day).mean()
    data["MAD"] = TP.rolling(window=day).apply(
        lambda x: np.mean(np.abs(x - np.mean(x)))
    )
    # data["CCI"] = (TP - data["SMA_TP"]) / (0.015 * data["MAD"])


def calculate_price_percentage_oscillator(data, day):  # Price Percentage Oscillator
    data["PPO"] = (data["EMA"] - data["EMA"].shift(day)) / data["EMA"].shift(day) * 100


def calculate_ichimoku_cloud(data, day):  # Ichimoku Cloud
    data["Tenkan_sen"] = (
        data["High"].rolling(window=day).max() + data["Low"].rolling(window=day).min()
    ) / 2
    data["Kijun_sen"] = (
        data["High"].rolling(window=day).max() + data["Low"].rolling(window=day).min()
    ) / 2
    data["Senkou_span_A"] = ((data["Tenkan_sen"] + data["Kijun_sen"]) / 2).shift(day)
    data["Senkou_span_B"] = (
        (data["High"].rolling(window=day).max() + data["Low"].rolling(window=day).min())
        / 2
    ).shift(day)


def create_features(data):
    calculate_ema(data, 14)
    calculate_bollinger_bands(data, 20)
    # calculate_rate_of_change(data, 14)
    calculate_commodity_channel_index(data, 14)
    # calculate_price_percentage_oscillator(data, 26)
    calculate_ichimoku_cloud(data, 9)
    data.dropna(inplace=True)
    return data
