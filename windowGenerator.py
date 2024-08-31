import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import uuid


class WindowGenerator:
    def __init__(
        self,
        input_width,
        label_width,
        shift,
        train_df,
        val_df,
        test_df,
        all_df,
        batc_size=32,
        label_columns=None,
    ):

        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.all_df = all_df

        self.batc_size = batc_size

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }

        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        self.test_df_date = self.test_df.index[self.input_width :]

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df, Shuffle=False)

    @property
    def test_shuffled(self):
        return self.make_dataset(self.test_df, Shuffle=True)

    @property
    def last_window(self):
        return self.make_test_dataset(
            self.test_df.iloc[:, -self.total_window_size :], Shuffle=False
        )

    @property
    def all(self):
        self.all_df_date = self.all_df.index[self.input_width :]
        return self.make_dataset(self.all_df, Shuffle=False)

    def __str__(self):
        return "\n".join(
            [
                f"Input width: {self.input_width}",
                f"Label width: {self.label_width}",
                f"Total window size: {self.total_window_size}",
                f"Shift: {self.shift}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}",
            ]
        )

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [
                    labels[:, :, self.column_indices[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.test_shuffled))  # burası train idi ben değiştirdim
            # And cache it for next time
            self._example = result
        return result

    @property
    def example_last_window(self):

        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.last_window))
            self._example = result
        return result

    def plot(
        self,
        model=None,
        model_name=None,
        date_term=None,
        model_main_folder=None,
        plot_col="Close",
        max_subplots=3,
    ):

        #
        print(self.test)
        #
        inputs, labels = self.example

        plt.figure(figsize=(12, 8))
        plt.title = f"{model_name} - {date_term}"
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f"{plot_col} [normed]")
            plt.plot(
                self.input_indices,
                inputs[n, :, plot_col_index],
                label="Inputs",
                marker=".",
                zorder=-10,
            )

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(
                self.label_indices,
                labels[n, :, label_col_index],
                edgecolors="k",
                label="Labels",
                c="#2ca02c",
                s=64,
            )

            if model is not None:
                predictions = model(inputs)
                plt.scatter(
                    self.label_indices,
                    predictions[n, :, label_col_index],
                    marker="X",
                    edgecolors="k",
                    label="Predictions",
                    c="#ff7f0e",
                    s=64,
                )

            if n == 0:
                plt.legend()

        plt.xlabel("Time")

        if (
            model_name is not None
            and date_term is not None
            and model_main_folder is not None
        ):
            self.save_plot(plt, model_main_folder)
        else:
            plt.show()

    def plot_last_window(
        self,
        model=None,
        model_name=None,
        date_term=None,
        plot_col="Close",
        max_subplots=1,
    ):
        inputs, labels = self.example_last_window

        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f"{plot_col} [normed]")
            plt.plot(
                self.input_indices,
                inputs[n, :, plot_col_index],
                label="Inputs",
                marker=".",
                zorder=-10,
            )

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(
                self.label_indices,
                labels[n, :, label_col_index],
                edgecolors="k",
                label="Labels",
                c="#2ca02c",
                s=64,
            )

            if model is not None:
                predictions = model(inputs)
                plt.scatter(
                    self.label_indices,
                    predictions[n, :, label_col_index],
                    marker="X",
                    edgecolors="k",
                    label="Predictions",
                    c="#ff7f0e",
                    s=64,
                )

            if n == 0:
                plt.legend()

        plt.xlabel("Time")

        if model_name is not None and date_term is not None:
            self.save_plot(plt, model_name)
        else:
            plt.show()

    def make_dataset(self, data, Shuffle=True):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=Shuffle,
            batch_size=self.batc_size,  # 32
        )

        ds = ds.map(self.split_window)

        return ds

    def make_test_dataset(self, data, Shuffle=False):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=Shuffle,
            batch_size=data.shape[0],
        )

        ds = ds.map(self.split_window)

        return ds

    def plot_test_labels(
        self, model, model_folder_path, scaler=None, model_name=None, date_term=None
    ):

        my_iter = iter(self.test)

        all_pred = None
        all_labels = None

        for _ in range(len(self.test)):
            inputs, labels = next(my_iter)
            new_pred = model(inputs)

            labels = labels[:, :: self.label_width, :]
            new_pred = new_pred[:, :: self.label_width, :]
            new_pred = tf.reshape(new_pred, (-1, 1))
            labels = tf.reshape(labels, (-1, labels.shape[-1]))

            all_pred = (
                new_pred
                if all_pred is None
                else tf.concat([all_pred, new_pred], axis=0)
            )
            all_labels = (
                labels
                if all_labels is None
                else tf.concat([all_labels, labels], axis=0)
            )

        # plot
        if scaler is not None:
            min_value = scaler.data_min_[3]  # Tahmin edilen sütun
            max_value = scaler.data_max_[3]  # Tahmin edilen sütun

            # Tahmin edilen değerleri orijinal boyutlarına döndürün
            predictions_original = all_pred * (max_value - min_value) + min_value
            labels_original = all_labels * (max_value - min_value) + min_value

            pred_go_up = np.where(
                predictions_original[1:] > predictions_original[:-1], 1, 0
            )
            labels_go_up = np.where(labels_original[1:] > labels_original[:-1], 1, 0)

            # Doğru tahminlerin sayısını hesaplama
            same_count = tf.reduce_sum(
                tf.cast(tf.equal(labels_go_up, pred_go_up), tf.float32)
            )

            print(f"Accuracy: {same_count/len(labels_go_up)}")
            rmse = np.sqrt(np.mean(((all_pred - all_labels) ** 2)))
            print(f"RMSE: {rmse}")

        plt.figure(figsize=(12, 8))
        plt.title = f"{model_name} - {date_term}"
        plt.plot(self.test_df_date, predictions_original, label="Predictions")
        plt.plot(self.test_df_date, labels_original, label="Labels")
        plt.legend()

        if model_name is not None and date_term is not None:
            self.save_plot(plt, model_main_folder=model_folder_path)
        else:
            plt.show()

    def predict_test_data_and_get_metrics(self, model, scaler=None):
        my_iter = iter(self.test)

        all_pred = None
        all_labels = None

        for _ in range(len(self.test)):
            inputs, labels = next(my_iter)
            new_pred = model(inputs)

            labels = labels[:, :: self.label_width, :]
            new_pred = new_pred[:, :: self.label_width, :]
            new_pred = tf.reshape(new_pred, (-1, 1))
            labels = tf.reshape(labels, (-1, labels.shape[-1]))

            all_pred = (
                new_pred
                if all_pred is None
                else tf.concat([all_pred, new_pred], axis=0)
            )
            all_labels = (
                labels
                if all_labels is None
                else tf.concat([all_labels, labels], axis=0)
            )

        # plot
        min_value = scaler.data_min_[3]  # Tahmin edilen sütun
        max_value = scaler.data_max_[3]  # Tahmin edilen sütun

        # Tahmin edilen değerleri orijinal boyutlarına döndürün
        predictions_original = all_pred * (max_value - min_value) + min_value
        labels_original = all_labels * (max_value - min_value) + min_value

        pred_go_up = np.where(
            predictions_original[1:] > predictions_original[:-1], 1, 0
        )
        labels_go_up = np.where(labels_original[1:] > labels_original[:-1], 1, 0)

        # Doğru tahminlerin sayısını hesaplama
        same_count = tf.reduce_sum(
            tf.cast(tf.equal(labels_go_up, pred_go_up), tf.float32)
        )

        accuracy = same_count / len(labels_go_up)
        rmse = np.sqrt(np.mean(((all_pred - all_labels) ** 2)))
        return accuracy, rmse

    def plot_all_data(self, model, model_name, date_term, model_folder_path):

        my_iter = iter(self.all)

        all_pred = None
        all_labels = None

        for _ in range(len(self.all)):
            inputs, labels = next(my_iter)
            new_pred = model(inputs)

            labels = labels[:, :: self.label_width, :]
            new_pred = new_pred[:, :: self.label_width, :]

            new_pred = tf.reshape(new_pred, (-1, 1))
            labels = tf.reshape(labels, (-1, labels.shape[-1]))

            all_pred = (
                new_pred
                if all_pred is None
                else tf.concat([all_pred, new_pred], axis=0)
            )
            all_labels = (
                labels
                if all_labels is None
                else tf.concat([all_labels, labels], axis=0)
            )

        # plot
        plt.figure(figsize=(12, 8))
        plt.title = f"{model_name} - {date_term}"
        plt.plot(self.all_df_date, all_pred, label="Predictions")
        plt.plot(self.all_df_date, all_labels, label="Labels")
        plt.legend()

        if model_name is not None and date_term is not None:
            self.save_plot(plt, model_main_folder=model_folder_path)
        else:
            plt.show()

    def save_plot(self, plot, model_main_folder):

        save_path = f"{model_main_folder}/plots/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        rand_name = str(uuid.uuid4())

        plot.savefig(save_path + f"/{rand_name}.png")
