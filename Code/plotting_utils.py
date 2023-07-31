import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns


# This normalizes the confusion matrix and ensures neat plotting for all outputs.
# Function for plotting confusion matrices


def plot_confusion_matrix(
    cm,
    classes,
    normalize=True,
    title="Confusion matrix",
    xrotation=0,
    yrotation=0,
    cmap=plt.cm.Purples,
    printout=False,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        if printout:
            print("Normalized confusion matrix")
    else:
        if printout:
            print("Confusion matrix")

    if printout:
        print(cm)

    plt.figure(figsize=(6, 4))

    plt.imshow(cm, interpolation="nearest", vmin=0.2, vmax=1.0, cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    classes = classes
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=xrotation)
    plt.yticks(tick_marks, classes, rotation=yrotation)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label", weight="bold")
    plt.xlabel("Predicted label", weight="bold")
    # plt.show()
    plt.savefig(
        (
            "C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\std_ML\Confusion_Matrix_"
            + figure_name
            + "_"
            + ".png"
        ),
        dpi=500,
        bbox_inches="tight",
    )


# Visualizing outputs
# for visualizing confusion matrix once the model is trained


def visualize(figure_name, classes, predicted, true):
    # Sort out predictions and true labels
    # for label_predictions_arr, label_true_arr, classes, outputs in zip(predicted, true, classes, outputs):
    #     print('visualize predicted classes', predicted)
    #     print('visualize true classes', true)
    classes_pred = np.asarray(predicted)
    classes_true = np.asarray(true)
    print(classes_pred.shape)
    print(classes_true.shape)
    classes = ["1-9", "10-16"]
    cnf_matrix = confusion_matrix(classes_true, classes_pred, labels=classes)
    plot_confusion_matrix(cnf_matrix, classes)
