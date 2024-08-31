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
    val_performance,
    performance,
    path,
    metric_name="binary_accuracy",  # mean_absolute_error
):
    plt.clf()
    plt.cla()

    for v in performance.values():
        print(v)

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
    model_folder_path,
) -> None:

    data_start_year = start_date.split("-")[0]

    model_folder_path = model_folder_path

    model.save(f"{model_folder_path}{model_name}.h5")

    val_performance[model_name] = model.evaluate(window.val, return_dict=True)
    performance[model_name] = model.evaluate(window.test, verbose=0, return_dict=True)

    pkl.dump(val_performance, open(f"{model_folder_path}/val_performance.pkl", "wb"))
    pkl.dump(performance, open(f"{model_folder_path}/performance.pkl", "wb"))
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
        gru_model, "GRU", data_start_year, model_folder_path=model_folder_path
    )

    plot_performance(
        model_name,
        data_start_year,
        val_performance,
        performance,
        model_folder_path,
        metric_name="binary_accuracy",
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


fm = FileManager("MSFT")

curr_folder_path = get_model_folder_path()

data = fm.processed_with_news_and_fred

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


long_term_wide_window = WindowGenerator(
    input_width=25,  # Adjust the input width to match the number of features in your data
    label_width=25,  # Adjust the label width to match the number of features in your data
    shift=1,
    train_df=long_train_df,
    val_df=long_val_df,
    test_df=long_test_df,
    all_df=long_term_df,
    batch_size=32,
    label_columns=["Close"],
)


long_term_wide_window.plot()
input("işlem tamam")


"""def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * 1.5  # Öğrenme hızını %5 artır


lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler)


reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=2, min_lr=0.00001
)
"""
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, mode="min", restore_best_weights=True
)

log_path = f"{curr_folder_path}logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_path, histogram_freq=1
)

callbacks = [early_stopping, tensorboard_callback]  #


def compile_and_fit(
    model: tf.keras.Model,
    window: WindowGenerator,
):

    model.compile(
        loss=tf.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),  # RMSprop
        metrics=[tf.metrics.BinaryAccuracy()],  # tf.metrics.BinaryAccuracy()
    )

    history = model.fit(
        window.train,
        epochs=200,
        validation_data=window.val,
        callbacks=callbacks,
    )

    return history


# define performance dictionary

val_performance = {}
performance = {}

# create models folder

if not os.path.exists("models"):
    os.makedirs("models")


# define GRU model

if os.path.exists("models/short_term_gru.pkl"):

    gru_model = pkl.load(open("models/short_term_gru.pkl", "rb"))
else:

    gru_model = tf.keras.models.Sequential(
        [
            tf.keras.layers.GRU(512, return_sequences=True),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units=1, activation="sigmoid")
            ),
        ]
    )


# long term

with tf.device("/GPU:0"):
    long_term_history = compile_and_fit(gru_model, long_term_wide_window)


log_model_info(
    gru_model,
    "GRU",
    start_date,
    end_date,
    long_term_wide_window,
    long_term_scaler,
    curr_folder_path,
)
