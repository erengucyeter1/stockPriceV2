import os
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

color_list = ["#1f77b4", "#ff7f0e", "#2ca02c"]
label_name_list = [
    "Baseline Models",
    "Models Incorporating News Titles",
    "Models Incorporating Full News Content",
]


val_measures = {}
label_list = []

first_models_path = "all_models/"
models_with_news_path = "models_sentiment/"
# models_with_content_path = "models_content/"

metric_name = "mean_absolute_error"

y_ticks_step = 0.005

paths = [first_models_path, models_with_news_path]  # , models_with_content_path

# pkl load

for j, path in enumerate(paths):
    folder_list = os.listdir(path)
    folder_list.sort(key=lambda x: os.path.getmtime(path + x))

    i = 0
    for folder in folder_list:

        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):
            continue
        with open(path + folder + "/performance.pkl", "rb") as f:
            test_mae = pkl.load(f)

            for v in test_mae.values():
                val_measures[f"{v}_{i}"] = v[metric_name]
                label_list.append(folder)
                i += 1

    x = np.arange(len(val_measures))
    width = 0.3
    test_mae = []

    for v in val_measures.values():
        test_mae.append(v)

    plt.ylabel(f"{metric_name} [Close]", fontsize=12)
    plt.yticks(np.arange(0, 0.3, y_ticks_step))

    plt.bar(
        x + 0.8,
        test_mae,
        width,
        label=label_name_list[j],
        color=color_list[j],
    )
    plt.xticks(ticks=x, labels=range(len(label_list)), rotation=45, fontsize=6)
    val_measures = {k: 0 for k in val_measures}


# plt.savefig(f"plots/{metric_name}_performance.png")
_ = plt.legend()

plt.show()

plt.cla()
plt.clf()
