import os
import pickle as pkl
import pandas as pd
from windowGenerator import WindowGenerator
from myData import GetData
from tqdm import tqdm

# for predict next day
"""
temp_row = data.tail(1).copy()
temp_row.iloc[:, 1:] = 0
data = pd.concat(
    [data, temp_row],
    axis=0,
)"""


main_path = "models_sentiment/"

model_folders = os.listdir(main_path)
model_folders.sort(key=lambda x: os.path.getmtime(main_path + x))


accuracy_list = []
rmse_list = []
model_name = []


if not os.path.exists(main_path + "accuracy_list.pkl"):
    data = pd.read_csv(
        "sentiment_processes/train_ready_data.csv"
    )  ## fred modeller için merged_data.csv

    long_n = len(data)

    data = data[int(long_n * 0.95) :]

    # process

    # scale data

    from sklearn.preprocessing import MinMaxScaler

    # short_term_scaler = MinMaxScaler()
    date_time = pd.to_datetime(data.pop("Date"), format="%Y-%m-%d")
    data.index = date_time

    # read sclaler

    with open(
        "C:\\Users\\ereng\\Desktop\\stockPriceV2\\models_sentiment\\models13_best_so_far\\long_term_scaler.pkl",
        "rb",
    ) as file:
        scaler = pkl.load(file)

    data = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)

    # short_term_df.index = short_date_time

    # create window generator

    wide_window = WindowGenerator(
        input_width=25,
        label_width=25,
        shift=1,
        train_df=data,
        val_df=data,
        test_df=data,
        all_df=data,
        label_columns=["Close"],
    )

    # predict

    for i, folder in tqdm(enumerate(model_folders)):

        model_path = main_path + folder + "/" + "long_term_gru.pkl"
        model_name.append(folder)
        model = pkl.load(open(model_path, "rb"))

        accuracy, rmse = wide_window.predict_test_data_and_get_metrics(model, scaler)
        accuracy_list.append(accuracy)
        rmse_list.append(rmse)

    # save accuracy and rmse

    with open(main_path + "accuracy_list.pkl", "wb") as file:
        pkl.dump(accuracy_list, file)

    with open(main_path + "rmse_list.pkl", "wb") as file:
        pkl.dump(rmse_list, file)
else:
    with open(main_path + "accuracy_list.pkl", "rb") as file:
        accuracy_list = pkl.load(file)

    with open(main_path + "rmse_list.pkl", "rb") as file:
        rmse_list = pkl.load(file)

    for i, folder in tqdm(enumerate(model_folders)):

        model_path = main_path + folder + "/" + "long_term_gru.pkl"
        model_name.append(folder)


# plot results


import matplotlib.pyplot as plt

plt.bar(
    range(len(accuracy_list)),
    accuracy_list,
    color="#1f77b4",
    width=0.4,
    label="Accuracy",
)
plt.bar(
    [x + 0.4 for x in range(len(rmse_list))],
    rmse_list,
    color="#ff7f0e",
    width=0.4,
    label="RMSE",
)
plt.xlabel("Model")
plt.ylabel("Metric")
plt.title("Accuracy and RMSE Comparison")
plt.xticks(
    [x + 0.2 for x in range(len(accuracy_list))],
    model_name[: len(accuracy_list)],
    rotation=90,
)
plt.legend()
plt.show()

plt.cla()
plt.clf()


# test one model

"""
data = pd.read_csv("sentiment_processes/train_ready_data.csv")

long_n = len(data)

data = data[int(long_n * 0.95) :]

# process

# scale data

from sklearn.preprocessing import MinMaxScaler

# short_term_scaler = MinMaxScaler()
date_time = pd.to_datetime(data.pop("Date"), format="%Y-%m-%d")
data.index = date_time

# read sclaler

with open(
    "C:\\Users\\ereng\\Desktop\\stockPriceV2\\models_sentiment\\models13_best_so_far\\long_term_scaler.pkl",
    "rb",
) as file:
    scaler = pkl.load(file)


data = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)

# short_term_df.index = short_date_time



# create window generator

wide_window = WindowGenerator(
    input_width=25,
    label_width=25,
    shift=1,
    train_df=data,
    val_df=data,
    test_df=data,
    all_df=data,
    label_columns=["Close"],
)
"""
