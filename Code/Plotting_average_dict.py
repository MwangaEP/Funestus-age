# %%
import json
import numpy as np

from matplotlib import pyplot
import matplotlib.pyplot as plt  # for making plots
import seaborn as sns

sns.set(
    context="paper",
    style="whitegrid",
    palette="deep",
    font_scale=2.0,
    color_codes=True,
    rc=({"font.family": "Helvetica"}),
)
# %matplotlib inline
plt.rcParams["figure.figsize"] = [6, 4]

# example_list_of_dicts = [{'loss': [60.901458740234375, 60.452842712402344, 59.98822784423828, 59.50740051269531, 59.02333450317383],
#                         'accuracy': [0.5192592740058899, 0.512592613697052, 0.529629647731781, 0.5192592740058899, 0.5355555415153503],
#                         'val_loss': [60.54882049560547, 60.1287841796875, 59.721778869628906, 59.26499938964844, 58.81584548950195],
#                         'val_accuracy': [0.5666666626930237, 0.47333332896232605, 0.4333333373069763, 0.4333333373069763, 0.4333333373069763]},
#                         {'loss': [60.2, 60.1, 59.98822784423828, 59.5, 59.0],
#                         'accuracy': [0.52, 0.522, 0.531, 0.519, 0.535],
#                         'val_loss': [60.54, 60.12, 59.72, 59.26, 58.81],
#                         'val_accuracy': [0.56, 0.47, 0.43, 0.43, 0.43]}]


# print(example_list_of_dicts )

# %%
# This takes a list of dictionaries, and combines them into a dictionary in which each key maps to a
# list of all the appropriate values from the parameter dictionaries


def combine_dictionaries(list_of_dictionaries):
    combined_dictionaries = {}

    for individual_dictionary in list_of_dictionaries:
        for key_value in individual_dictionary:
            if key_value not in combined_dictionaries:
                combined_dictionaries[key_value] = []
            combined_dictionaries[key_value].append(individual_dictionary[key_value])

    return combined_dictionaries


# %%

# right now, no error checking, assumed all lists are of same length
# def find_mean_from_combined_dicts(combined_dicts):

#     dict_of_means = {}

#     for key_value in combined_dicts:
#         dict_of_means[key_value] = []

#         for i in range(len(combined_dicts[key_value][0])):
#             this_index_values = []
#             for j in range(len(combined_dicts[key_value])):
#                 this_index_values.append(combined_dicts[key_value][j][i])
#             mean_value = sum(this_index_values)/len(this_index_values)
#             dict_of_means[key_value].append(mean_value)

#     return dict_of_means


def find_mean_from_combined_dicts(combined_dicts):
    dict_of_means = {}

    for key_value in combined_dicts:
        dict_of_means[key_value] = []

        # Length of longest list return the longest list within the list of a dictionary item
        length_of_longest_list = max([len(a) for a in combined_dicts[key_value]])
        temp_array = np.empty([len(combined_dicts[key_value]), length_of_longest_list])
        temp_array[:] = np.NaN

        for i, j in enumerate(combined_dicts[key_value]):
            temp_array[i][0 : len(j)] = j
        mean_value = np.nanmean(temp_array, axis=0)

        dict_of_means[key_value] = mean_value.tolist()

    return dict_of_means


# %%


def smooth_curve(points, factor=0.95):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def set_plot_history_data(ax, history, which_graph):
    if which_graph == "accuracy":
        train = smooth_curve(history["accuracy"])
        valid = smooth_curve(history["val_accuracy"])

    epochs = range(1, len(train) + 1)

    trim = 5  # remove first n epochs
    # when graphing loss the first few epochs may skew the (loss) graph

    ax.plot(epochs[trim:], train[trim:], "b", label=("accuracy"))
    ax.plot(epochs[trim:], train[trim:], "b", linewidth=10, alpha=0.1)

    ax.plot(epochs[trim:], valid[trim:], "orange", label=("val_accuracy"))
    ax.plot(epochs[trim:], valid[trim:], "orange", linewidth=10, alpha=0.1)


def graph_history_averaged(combined_history):
    print("averaged_histories.keys : {}".format(combined_history.keys()))
    fig, (ax1) = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(6, 4),
        sharex=True,
    )

    set_plot_history_data(ax1, combined_history, "accuracy")

    # Accuracy graph
    ax1.set_ylabel("Accuracy", weight="bold")
    plt.xlabel("Epoch", weight="bold")
    # ax1.set_ylim(bottom = 0.3, top = 1.0)
    ax1.legend(loc="lower right")
    # ax1.set_yticks(np.arange(0.3, 1.0 + .2, step = 0.1))
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.xaxis.set_ticks_position("bottom")
    ax1.spines["bottom"].set_visible(True)

    plt.yticks(np.arange(0.2, 1.0 + 0.1, step=0.2))
    plt.xticks(range(0, len(combined_history["accuracy"]) + 500, 2000))
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(
        "C:/Mannu/Projects/Anophles Funestus Age Grading (WILD)/Fold/Training_Folder_selected_wns/Averaged_graph_2.png",
        dpi=500,
        bbox_inches="tight",
    )
    plt.close()


# %%

with open(
    "C:/Mannu/Projects/Anophles Funestus Age Grading (WILD)/Fold/Training_Folder_selected_wns/combined_history_dictionaries.txt"
) as json_file:
    combined_dictionary = json.load(json_file)

# %%

combined_avrg = find_mean_from_combined_dicts(combined_dictionary)
graph_history_averaged(combined_avrg)

# %%
