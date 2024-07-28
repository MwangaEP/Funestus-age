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
    output,
    save_path,
    model_name,
    fold,
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
    # plt.title([title +' - '+ model_name])
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
    plt.ylabel("True Age", weight="bold")
    plt.xlabel("Predicted Age", weight="bold")


def visualizeML(figure_name, save_path, classes, predicted, true, model_name, fold):
    # Sort out predictions and true labels
    # for label_predictions_arr, label_true_arr, classes, outputs in zip(predicted, true, classes, outputs):
    #     print('visualize predicted classes', predicted)
    #     print('visualize true classes', true)

    classes_pred = np.asarray(predicted)
    classes_true = np.asarray(true)
    classes = classes
    print(classes_pred.shape)
    print(classes_true.shape)
    cnf_matrix = confusion_matrix(classes_true, classes_pred, labels=classes)
    plot_confusion_matrix(cnf_matrix, classes, save_path, figure_name, model_name, fold)
    plt.savefig(
        (save_path + "Confusion_Matrix_" + figure_name + "_" + ".png"),
        dpi=500,
        bbox_inches="tight",
    )


def visualizeDL(
    histories, save_path, model_name, fold, classes, outputs, predicted, true
):
    # Sort out predictions and true labels
    # for label_predictions_arr, label_true_arr, classes, outputs in zip(predicted, true, classes, outputs):
    # print('visualize predicted classes', predicted)
    # print('visualize true classes', true)

    classes_pred = np.argmax(predicted, axis=-1)
    classes_true = np.argmax(true, axis=-1)
    print(classes_pred.shape)
    print(classes_true.shape)
    cnf_matrix = confusion_matrix(classes_true, classes_pred)
    plot_confusion_matrix(cnf_matrix, classes, outputs, save_path, model_name, fold)
    plt.savefig(
        (save_path + "Confusion_Matrix_" + model_name + "_" + fold + "_" + ".png"),
        dpi=500,
        bbox_inches="tight",
    )
