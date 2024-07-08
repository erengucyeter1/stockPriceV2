from myData import GetData
import pandas as pd
from windowGenerator import WindowGenerator

import numpy as np
import tensorflow as tf

"""## dense ile güzel sonuçlar alındı
GetData = GetData("AAPL", "2020-06-01", "2024-07-01")"""

GetData = GetData("MSFT", "2020-06-01", "2024-07-01")


"""
feate extracted data
Validation MAE: 0.06132841110229492
Validation Loss: 0.005978615954518318"""

"""
raw data without volume
Validation MAE: 0.053314365446567535
Validation Loss: 0.004720442928373814"""

"""
raw data with volume
Validation MAE: 0.05568355321884155
Validation Loss: 0.005070376675575972
"""


df = GetData.get_raw_data()

df.pop("Volume")


# scale data

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])


date_time = pd.to_datetime(df.pop("Date"), format="%Y-%m-%d")

df.index = date_time


n = len(df)
train_df = df[0 : int(n * 0.7)]
val_df = df[int(n * 0.7) : int(n * 0.9)]
test_df = df[int(n * 0.9) :]

num_features = df.shape[1]

train_mean = train_df.mean()
train_std = train_df.std()


train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std


"""##dense de iyi oldu
wide_window = WindowGenerator(
    input_width=60,
    label_width=60,
    shift=1,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    all_df=df,
    label_columns=["Close"],
)"""

# lstm için
wide_window = WindowGenerator(
    input_width=10,
    label_width=10,
    shift=1,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    all_df=df,
    label_columns=["Close"],
)

print(repr(wide_window))

# models


def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min"
    )

    model.compile(
        loss=tf.losses.MeanSquaredError(),
        # optimizer=tf.optimizers.Adam(),
        optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
        metrics=[tf.metrics.MeanAbsoluteError()],
    )

    history = model.fit(
        window.train,
        epochs=60,
        validation_data=window.val,
        callbacks=[early_stopping],
    )

    return history


# dense model

"""dense = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(units=64, activation="relu"),
        tf.keras.layers.Dense(units=64, activation="relu"),
        tf.keras.layers.Dense(units=1),
    ]
)

history = compile_and_fit(dense, wide_window)

val_performance = dense.evaluate(wide_window.val, return_dict=True)
performance = dense.evaluate(wide_window.test, verbose=0, return_dict=True)

print(f"Validation MAE: {val_performance['mean_absolute_error']}")
print(f"Validation Loss: {val_performance['loss']}")

# wide_window.plot(dense)
wide_window.plot_test_labels(dense)"""


"""Validation MAE: 0.07145874202251434
Validation Loss: 0.008566766045987606"""


# dense 2

# dense model
"""
dense = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(units=256, activation="relu"),
        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dense(units=256, activation="relu"),
        tf.keras.layers.Dense(units=1),
    ]
)

history = compile_and_fit(dense, wide_window)

val_performance = dense.evaluate(wide_window.val, return_dict=True)
performance = dense.evaluate(wide_window.test, verbose=0, return_dict=True)

print(f"Validation MAE: {val_performance['mean_absolute_error']}")
print(f"Validation Loss: {val_performance['loss']}")

##wide_window.plot(dense, max_subplots=5)

# wide_window.plot_test_labels(dense)
wide_window.plot_all_data(dense)"""

# lstm model

"""lstm_model = tf.keras.models.Sequential(
    [
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(64, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1),
    ]
)


print("Input shape:", wide_window.example[0].shape)
print("Output shape:", lstm_model(wide_window.example[0]).shape)

history = compile_and_fit(lstm_model, wide_window)

val_performance = lstm_model.evaluate(wide_window.val, return_dict=True)
performance = lstm_model.evaluate(wide_window.test, verbose=0, return_dict=True)

print(f"Validation MAE: {val_performance['mean_absolute_error']}")
print(f"Validation Loss: {val_performance['loss']}")

##wide_window.plot(lstm_model)
wide_window.plot_all_data(lstm_model)"""

"""Validation MAE: 0.11135562509298325
Validation Loss: 0.02502657100558281"""


"""OUT_STEPS = 10
multi_window = WindowGenerator(
    input_width=24,
    label_width=OUT_STEPS,
    shift=OUT_STEPS,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    all_df=df,
)"""

"""multi_lstm_model = tf.keras.Sequential(
    [
        # Shape [batch, time, features] => [batch, lstm_units].
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(32, return_sequences=False),
        # Shape => [batch, out_steps*features].
        tf.keras.layers.Dense(
            OUT_STEPS * num_features, kernel_initializer=tf.initializers.zeros()
        ),
        # Shape => [batch, out_steps, features].
        tf.keras.layers.Reshape([OUT_STEPS, num_features]),
    ]
)

history = compile_and_fit(multi_lstm_model, multi_window)


multi_val_performance = multi_lstm_model.evaluate(multi_window.val)
multi_performance = multi_lstm_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_lstm_model)


print(f"Validation MAE: {multi_val_performance[1]}")
print(f"Validation Loss: {multi_val_performance[0]}")"""

# multi step dense

"""multi_dense_model = tf.keras.Sequential(
    [
        # Take the last time step.
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # Shape => [batch, 1, dense_units]
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(512, activation="relu"),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(
            OUT_STEPS * num_features, kernel_initializer=tf.initializers.zeros()
        ),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features]),
    ]
)

history = compile_and_fit(multi_dense_model, multi_window)


multi_val_performance = multi_dense_model.evaluate(multi_window.val)
multi_performance = multi_dense_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_dense_model)"""


# GRU

gru_model = tf.keras.models.Sequential(
    [
        tf.keras.layers.GRU(64, return_sequences=True, activation="tanh"),
        tf.keras.layers.GRU(64, return_sequences=True, activation="tanh"),
        tf.keras.layers.Dense(units=1),
    ]
)

history = compile_and_fit(gru_model, wide_window)

val_performance = gru_model.evaluate(wide_window.val, return_dict=True)
performance = gru_model.evaluate(wide_window.test, verbose=0, return_dict=True)

print(f"Validation MAE: {val_performance['mean_absolute_error']}")
print(f"Validation Loss: {val_performance['loss']}")

wide_window.plot(gru_model)
wide_window.plot_all_data(gru_model)
