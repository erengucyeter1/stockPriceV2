import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import pandas as pd
from windowGenerator import WindowGenerator
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_manager import FileManager
import pickle as pkl
import os
from datetime import datetime


def plot_performance(
    model_name,
    date_term,
    performance,
    val_performance,
    path,
    metric_name="mean_absolute_error",  # mean_absolute_error
):
    plt.clf()
    plt.cla()

    x = np.arange(len(performance))
    width = 0.3
    val_mae = [v[metric_name] for v in val_performance.values()]
    test_mae = [v[metric_name] for v in performance.values()]

    plt.title = f"{model_name} - {date_term}"
    plt.ylabel(f"{metric_name} [Close, normalized]")
    plt.bar(x - 0.17, val_mae, width, label="Validation")
    plt.bar(x + 0.17, test_mae, width, label="Test")
    plt.xticks(ticks=x, labels=performance.keys(), rotation=45, fontsize=6)
    _ = plt.legend()
    if not os.path.exists(f"{path}plots"):
        os.makedirs(f"{path}plots")
    plt.savefig(f"{path}plots/{metric_name}_performance.png")
    pkl.dump(performance, open(f"{path}performance.pkl", "wb"))
    pkl.dump(val_performance, open(f"{path}val_performance.pkl", "wb"))


def get_model_folder_path(main_model_folder_path: str = "models/"):
    model_count = 0
    model_folder_path = f"{main_model_folder_path}model_{model_count+1}/"

    if not os.path.exists(main_model_folder_path):
        os.makedirs(main_model_folder_path)
        os.makedirs(model_folder_path)
    else:
        model_count = len(os.listdir(main_model_folder_path))
        model_folder_path = f"{main_model_folder_path}model_{model_count+1}/"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

    return model_folder_path


def log_model_info(
    model: tf.keras.Model,
    model_name: str,
    start_date: str,
    end_date: str,
    window: WindowGenerator,
    scaler: MinMaxScaler,
    performance,
    val_performance,
    model_folder_path,
) -> None:

    data_start_year = start_date.split("-")[0]

    model_folder_path = model_folder_path

    model.save(f"{model_folder_path}{model_name}.h5")

    pkl.dump(scaler, open(f"{model_folder_path}/scaler.pkl", "wb"))

    window.plot(
        model=model,
        model_name=model_name,
        model_main_folder=model_folder_path,
        date_term=data_start_year,
    )
    window.plot_test_labels(
        model=model,
        scaler=scaler,
        model_name=model_name,
        date_term=data_start_year,
        model_folder_path=model_folder_path,
    )
    window.plot_all_data(
        model, "GRU", data_start_year, model_folder_path=model_folder_path
    )

    plot_performance(
        model_name, data_start_year, performance, val_performance, model_folder_path
    ),

    model_info = open(f"{model_folder_path}model_info.txt", "a")
    model.summary(print_fn=lambda x: model_info.write(x + "\n"))
    model_info.write("\n\n")
    model_info.write(f"Model Name: {model_name}\n")
    model_info.write(f"Train data start Date: {start_date}\n")
    model_info.write(f"Train data end Date: {end_date}\n")
    model_info.write(f"Window Generator: {window}\n")
    model_info.write("\n\n")
    model_info.write("Model Hyperparameters:\n")
    model_info.write(f"Learning Rate: {model.optimizer.learning_rate.numpy()}\n")
    model_info.write(f"Batch Size: {window.batch_size}\n")
    model_info.write("\n\n")
    model_info.write(f"Validation Performance: {val_performance[model_name]}\n")
    model_info.write(f"Test Performance: {performance[model_name]}\n")
    model_info.write("\n\n")
    model_info.close()

    del model
    # rmse ve accuracy eklenecek


fm = FileManager("MSFT")

curr_folder_path = get_model_folder_path()

data = fm.processed_with_news_and_fred.iloc[600:]

start_date = data["Date"].iloc[0]


# drop rows with NaN values
data.dropna(inplace=True)


long_term_scaler = MinMaxScaler()


data.iloc[:, 1:] = long_term_scaler.fit_transform(data.iloc[:, 1:])


data.index = pd.to_datetime(data.pop("Date"), format="%Y-%m-%d")


long_term_df = data

# normalize data

data = long_term_scaler.transform(data)


# split data


long_n = len(long_term_df)
long_train_df = long_term_df[0 : int(long_n * 0.7)]
long_val_df = long_term_df[int(long_n * 0.7) : int(long_n * 0.9)]
long_test_df = long_term_df[int(long_n * 0.9) :]

end_date = long_train_df.index[-1].strftime("%Y-%m-%d")
# create window generator


# hparam tuning


from tensorboard.plugins.hparams import api as hp
import gc

HP_WINDOW_SIZE = hp.HParam("window_size", hp.Discrete([15, 20, 25, 30, 35, 40]))
HP_BATCH_SIZE = hp.HParam("batch_size", hp.Discrete([32, 64, 128, 256]))
HP_NUM_UNITS = hp.HParam("num_units", hp.Discrete([64, 128, 256, 512]))
HP_DROPOUT = hp.HParam("dropout", hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adam", "RMSProp"]))

METRIC_ACCURACY = "mean_absolute_error"

with tf.summary.create_file_writer("logs/hparam_tuning").as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name="Mean Squared Error")],
    )


def train_test_model(hparams, window, logdir, layer_count=1):

    model = tf.keras.models.Sequential()

    for _ in range(layer_count):
        model.add(tf.keras.layers.GRU(hparams[HP_NUM_UNITS], return_sequences=True))
    model.add(tf.keras.layers.Dropout(hparams[HP_DROPOUT]))
    model.add(tf.keras.layers.Dense(1))

    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss=tf.losses.MeanAbsoluteError(),
        metrics=["mean_absolute_error"],
    )

    model.fit(
        window.train,
        epochs=15,
        callbacks=[
            tf.keras.callbacks.TensorBoard(logdir),  # log metrics
            hp.KerasCallback(logdir, hparams),  # log hparams
        ],
    )  # Run with 1 epoch to speed things up for demo purposes
    val_performance = {}
    performance = {}

    val_performance["GRU"] = model.evaluate(window.val, return_dict=True)
    performance["GRU"] = model.evaluate(window.test, verbose=0, return_dict=True)

    log_model_info(
        model,
        "GRU",
        start_date,
        end_date,
        window,
        long_term_scaler,
        performance=performance,
        val_performance=val_performance,
        model_folder_path=get_model_folder_path(),
    )
    return performance["GRU"]["mean_absolute_error"]


def run(run_dir, hparams, layer_count):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        performance = train_test_model(
            hparams, long_term_wide_window, run_dir, layer_count
        )
        tf.summary.scalar(METRIC_ACCURACY, performance, step=1)


session_num = 0
with tf.device("/GPU:0"):

    for layer_count in range(1, 4):
        for batch_size in HP_BATCH_SIZE.domain.values:
            for window_size in HP_WINDOW_SIZE.domain.values:

                long_term_wide_window = WindowGenerator(
                    input_width=window_size,
                    label_width=window_size,
                    shift=1,
                    train_df=long_train_df,
                    val_df=long_val_df,
                    test_df=long_test_df,
                    all_df=long_term_df,
                    batch_size=batch_size,
                    label_columns=["Close"],
                )

                for num_units in HP_NUM_UNITS.domain.values:
                    for dropout_rate in (
                        HP_DROPOUT.domain.min_value,
                        HP_DROPOUT.domain.max_value,
                    ):
                        for optimizer in HP_OPTIMIZER.domain.values:

                            if session_num < 352:
                                session_num += 1
                                continue

                            hparams = {
                                HP_NUM_UNITS: num_units,
                                HP_DROPOUT: dropout_rate,
                                HP_OPTIMIZER: optimizer,
                            }
                            run_name = "run-%d" % session_num
                            print("--- Starting trial: %s" % run_name)
                            print({h.name: hparams[h] for h in hparams})
                            tf.keras.backend.clear_session()  # Clear TensorFlow session and release GPU memory
                            run("logs/hparam_tuning/" + run_name, hparams, layer_count)
                            session_num += 1
                            gc.collect()

                del long_term_wide_window
