import os
import pickle
import pandas as pd

val_performance = {}
performance = {}


main_path = "HPARAM_MODELS/"


for folder in os.listdir(main_path):

    performance_path = f"{main_path}{folder}/performance.pkl"
    val_performance_path = f"{main_path}{folder}/val_performance.pkl"

    if not os.path.exists(performance_path) or not os.path.exists(val_performance_path):
        continue

    performance[folder] = pickle.load(open(performance_path, "rb"))
    val_performance[folder] = pickle.load(open(val_performance_path, "rb"))


# plot_performance


import matplotlib.pyplot as plt

x_labels = []

test_list = []
val_list = []

for model_name in performance.keys():

    test = performance[model_name]
    val = val_performance[model_name]

    test_list.append(test["GRU"]["mean_absolute_error"])
    val_list.append(val["GRU"]["mean_absolute_error"])
    x_labels.append(model_name)

df = pd.DataFrame({"Test": test_list, "Validation": val_list}, index=x_labels)

df.sort_values(by="Test", inplace=True)

df.to_csv(f"{main_path}performance.csv")

plt.figure(figsize=(20, 7))
plt.plot(df["Test"], label="Test")
plt.plot(df["Validation"], label="Validation")
plt.xticks(rotation=45)
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.title("Performance Comparison")
plt.legend()
plt.savefig("performance_plot.png")
plt.show()
