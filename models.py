import os

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

from windowGenerator import WindowGenerator
from myData import GetData

from sklearn.preprocessing import MinMaxScaler

# get data

ticker = "MSFT"

get_data = GetData(ticker, "2020-06-01", "2024-07-01")

df = get_data.get_raw_data()
