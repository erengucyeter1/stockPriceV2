from myData import GetData
import pandas as pd
from windowGenerator import WindowGenerator

import numpy as np
import tensorflow as tf

GetData = GetData("MSFT", "2022-01-01", "2024-07-01")

df = GetData.getData()

date_time = pd.to_datetime(df.pop("Date"), format="%Y-%m-%d")


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


"""w2 = WindowGenerator(
    input_width=6,
    label_width=1,
    shift=1,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=["Close"],
)

print(repr(w2))


# Stack three slices, the length of the total window.
example_window = tf.stack(
    [
        np.array(train_df[: w2.total_window_size]),
        np.array(train_df[100 : 100 + w2.total_window_size]),
        np.array(train_df[200 : 200 + w2.total_window_size]),
    ]
)

example_inputs, example_labels = w2.split_window(example_window)

w2.set_example(example_inputs, example_labels)


print("All shapes are: (batch, time, features)")
print(f"Window shape: {example_window.shape}")
print(f"Inputs shape: {example_inputs.shape}")
print(f"Labels shape: {example_labels.shape}")

w2.plot()


for example_inputs, example_labels in w2.train.take(1):
    print(f"Inputs shape (batch, time, features): {example_inputs.shape}") # daha iyi anlamaya çalış
    print(f"Labels shape (batch, time, features): {example_labels.shape}")"""


wide_window = WindowGenerator(
    input_width=20,
    label_width=20,
    shift=1,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
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
        optimizer=tf.optimizers.Adam(),
        metrics=[tf.metrics.MeanAbsoluteError()],
    )

    history = model.fit(
        window.train,
        epochs=60,
        validation_data=window.val,
        callbacks=[early_stopping],
    )

    return history


# one step model
"""
linear = tf.keras.Sequential([tf.keras.layers.Dense(units=1)])

history = compile_and_fit(linear, wide_window)

val_performance = linear.evaluate(wide_window.val, return_dict=True) ?
performance = linear.evaluate(wide_window.test, verbose=0, return_dict=True) ?

print(f"Validation MAE: {val_performance['mean_absolute_error']}") ?
print(f"Validation Loss: {val_performance['loss']}") ?

wide_window.plot(linear)
"""

# dense model

dense = tf.keras.Sequential(
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

wide_window.plot(dense)

# multi step dense model
CONV_WIDTH = 3
"""
conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=["Close"],
)"""

"""


multi_step_dense = tf.keras.Sequential(
    [
        # Shape: (time, features) => (time*features)
        tf.keras.layers.Flatten(),  # flatten nedir araştır
        tf.keras.layers.Dense(units=32, activation="relu"),  # dense nedir araştır
        tf.keras.layers.Dense(units=32, activation="relu"),
        tf.keras.layers.Dense(units=1),
        # Add back the time dimension.
        # Shape: (outputs) => (1, outputs)
        tf.keras.layers.Reshape([1, -1]),
    ]
)


history = compile_and_fit(multi_step_dense, conv_window)

val_performance = multi_step_dense.evaluate(conv_window.val, return_dict=True)
performance = multi_step_dense.evaluate(conv_window.test, verbose=0, return_dict=True)

print(f"Validation MAE: {val_performance['mean_absolute_error']}")
print(f"Validation Loss: {val_performance['loss']}")

conv_window.plot(multi_step_dense)  # bu kısmda bir not var sitede dense_4 yazan oku

"""

# cnn model


# conv_window açılmalı
"""conv_model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv1D(filters=32, kernel_size=(CONV_WIDTH), activation="relu"),
        tf.keras.layers.Dense(units=32, activation="relu"),
        tf.keras.layers.Dense(units=1),
    ]
)
"""
"""print("Conv model on `conv_window`")
print("Input shape:", conv_window.example[0].shape)
print("Output shape:", conv_model(conv_window.example[0]).shape)

history = compile_and_fit(conv_model, conv_window)

val_performance = conv_model.evaluate(conv_window.val, return_dict=True)
performance = conv_model.evaluate(conv_window.test, verbose=0, return_dict=True)

conv_window.plot(conv_model)"""

# wide_conv_window
"""CONV_WIDTH = 3  # eskiden de tanımlıydı

LABEL_WIDTH = 24

INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)

wide_conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=["Close"],
)

print("Wide conv window")
print("Input shape:", wide_conv_window.example[0].shape)
print("Labels shape:", wide_conv_window.example[1].shape)
print("Output shape:", conv_model(wide_conv_window.example[0]).shape)

wide_conv_window.plot(conv_model)"""


# lstm model

"""lstm_model = tf.keras.models.Sequential(
    [
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
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

wide_window.plot(lstm_model)
wide_window.plot(lstm_model)
"""


# modellerin karşılaştırmasının stun grafiği var oraya bak


# multi output models
"""wide_window = WindowGenerator(
    input_width=24,
    label_width=24,
    shift=1,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=["Close"],
)

lstm_model = tf.keras.models.Sequential(
    [
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=num_features),
    ]
)

history = compile_and_fit(lstm_model, wide_window)


val_performance = lstm_model.evaluate(wide_window.val, return_dict=True)
performance = lstm_model.evaluate(wide_window.test, verbose=0, return_dict=True)

wide_window.plot(lstm_model, plot_col="Close")
"""

# ??? multi output model çıkışları tam çözemedim diğer stunları nasıl çıkartıyor
"""


class ResidualWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)

        # The prediction for each time step is the input
        # from the previous time step plus the delta
        # calculated by the model.
        return inputs + delta


residual_lstm = ResidualWrapper(
    tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dense(
                num_features,
                # The predicted deltas should start small.
                # Therefore, initialize the output layer with zeros.
                kernel_initializer=tf.initializers.zeros(),
            ),
        ]
    )
)

history = compile_and_fit(residual_lstm, wide_window)


val_performance = residual_lstm.evaluate(wide_window.val, return_dict=True)
performance = residual_lstm.evaluate(wide_window.test, verbose=0, return_dict=True)

print(f"Validation MAE: {val_performance['mean_absolute_error']}")
print(f"Validation Loss: {val_performance['loss']}")

wide_window.plot(residual_lstm, plot_col="Close")"""


## multi step models

"""OUT_STEPS = 24

multi_window = WindowGenerator(
    input_width=24,
    label_width=OUT_STEPS,
    shift=OUT_STEPS,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
)"""
"""

# çok kötü çalışıyor



multi_lstm_model = tf.keras.Sequential(
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


multi_val_performance = multi_lstm_model.evaluate(multi_window.val, return_dict=True)
multi_performance = multi_lstm_model.evaluate(
    multi_window.test, verbose=0, return_dict=True
)
multi_window.plot(multi_lstm_model)"""


# RNN
"""

class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)


feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)


def warmup(self, inputs):
    # inputs.shape => (batch, time, features)
    # x.shape => (batch, lstm_units)
    x, *state = self.lstm_rnn(inputs)

    # predictions.shape => (batch, features)
    prediction = self.dense(x)
    return prediction, state


FeedBack.warmup = warmup

prediction, state = feedback_model.warmup(multi_window.example[0])


def call(self, inputs, training=None):
    # Use a TensorArray to capture dynamically unrolled outputs.
    predictions = []
    # Initialize the LSTM state.
    prediction, state = self.warmup(inputs)

    # Insert the first prediction.
    predictions.append(prediction)

    # Run the rest of the prediction steps.
    for n in range(1, self.out_steps):
        # Use the last prediction as input.
        x = prediction
        # Execute one lstm step.
        x, state = self.lstm_cell(x, states=state, training=training)
        # Convert the lstm output to a prediction.
        prediction = self.dense(x)
        # Add the prediction to the output.
        predictions.append(prediction)

    # predictions.shape => (time, batch, features)
    predictions = tf.stack(predictions)
    # predictions.shape => (batch, time, features)
    predictions = tf.transpose(predictions, [1, 0, 2])
    return predictions


FeedBack.call = call


history = compile_and_fit(feedback_model, multi_window)


multi_val_performance = feedback_model.evaluate(multi_window.val, return_dict=True)
multi_performance = feedback_model.evaluate(
    multi_window.test, verbose=0, return_dict=True
)
multi_window.plot(feedback_model)"""
