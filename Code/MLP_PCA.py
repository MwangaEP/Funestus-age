# This program uses deep learning trained with PCA transformed data to predict the age structure of of Anopheles funestus mosquitoes collected from the wild

# %%

import os
import io
import ast
import itertools
import collections
from time import time
from tqdm import tqdm

from itertools import cycle
import pickle
import datetime
import json

import numpy as np
import pandas as pd

import random as rn
from random import randint

from collections import Counter

from sklearn.model_selection import (
    ShuffleSplit,
    train_test_split,
    StratifiedKFold,
    StratifiedShuffleSplit,
    KFold,
)
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.preprocessing import (
    MultiLabelBinarizer,
    FunctionTransformer,
    LabelBinarizer,
)
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    recall_score,
    precision_score,
)

from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    f_classif,
    chi2,
    mutual_info_classif,
)

from plotting_utils import visualizeDL

from sklearn import decomposition
from sklearn.manifold import MDS

from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras import initializers
from keras.models import Sequential, Model
from keras import layers, metrics
from keras.layers import Input
from keras.layers import Concatenate
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import BatchNormalization
from keras.layers import Conv1D, MaxPooling1D
from keras.models import model_from_json, load_model
from keras.regularizers import *
from keras.callbacks import CSVLogger

# from tensorflow.keras import backend as K

import matplotlib.pyplot as plt  # for making plots
import seaborn as sns

sns.set(
    context="paper",
    style="white",
    palette="deep",
    font_scale=2.0,
    color_codes=True,
    rc=None,
)
# %matplotlib inline
plt.rcParams["figure.figsize"] = [6, 4]

# %%

# Upload An. funestus train data for model training

train_data = pd.read_csv("../Data/train_an_fun_df.csv")

print(train_data.head())

print(train_data.shape)

# Checking class distribution in the data
print(Counter(train_data["Cat3"]))

# drops columns of no interest
train_data = train_data.drop(["Unnamed: 0"], axis=1)
train_data.head(10)


# %%

# create a new folder for the CNN outputs


def build_folder(Fold, to_build=False):
    if not os.path.isdir(Fold):
        if to_build == True:
            os.mkdir(Fold)
        else:
            print("Directory does not exists, not creating directory!")
    else:
        if to_build == True:
            raise NameError("Directory already exists, cannot be created!")


# %%
# Data logging
# for logging data associated with the model


def log_data(log, name, fold, save_path):
    f = open((save_path + name + "_" + str(fold) + "_log.txt"), "w")
    np.savetxt(f, log)
    f.close()


# %%

# Graphing the training data and validation


def graph_history(history, model_name, model_ver_num, fold, save_path):
    # not_validation = list(filter(lambda x: x[0:3] != "val", history.history.keys()))
    print("history.history.keys : {}".format(history.history.keys()))
    filtered = filter(lambda x: x[0:3] != "val", history.history.keys())
    not_validation = list(filtered)
    for i in not_validation:
        plt.figure(figsize=(6, 4))
        # plt.title(i+"/ "+"val_"+i)
        plt.plot(history.history[i], label=i)
        plt.plot(history.history["val_" + i], label="val_" + i)
        plt.legend()
        plt.tight_layout()
        plt.grid(False)
        plt.xlabel("epoch", weight="bold")
        plt.ylabel(i)
        # plt.savefig(
        #     save_path
        #     + model_name
        #     + "_"
        #     + str(model_ver_num)
        #     + "_"
        #     + str(fold)
        #     + "_"
        #     + i
        #     + ".png",
        #     dpi=500,
        #     bbox_inches="tight",
        # )
        # plt.savefig(
        #     save_path
        #     + model_name
        #     + "_"
        #     + str(model_ver_num)
        #     + "_"
        #     + str(fold)
        #     + "_"
        #     + i
        #     + ".pdf",
        #     dpi=500,
        #     bbox_inches="tight",
        # )
        plt.close()


# Graphing the averaged training and validation histories when plotting, smooth out the points by some factor (0.5 = rough, 0.99 = smooth)
# method learned from `Deep Learning with Python` by François Chollet


def smooth_curve(points, factor=0.75):
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

    trim = 0  # remove first 5 epochs
    # when graphing loss the first few epochs may skew the (loss) graph

    ax.plot(epochs[trim:], train[trim:], "b", label=("accuracy"))
    ax.plot(epochs[trim:], train[trim:], "b", linewidth=15, alpha=0.1)

    ax.plot(epochs[trim:], valid[trim:], "orange", label=("val_accuracy"))
    ax.plot(epochs[trim:], valid[trim:], "orange", linewidth=15, alpha=0.1)


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
    ax1.set_yticks(np.arange(0.2, 1.0 + 0.1, step=0.1))
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.xaxis.set_ticks_position("bottom")
    ax1.spines["bottom"].set_visible(True)

    plt.tight_layout()
    plt.grid(False)
    plt.savefig(
        "../Results/MLP_pca\Training_Folder\Averaged_graph.png",
        dpi=500,
        bbox_inches="tight",
    )
    plt.close()


# %%
# This takes a list of dictionaries, and combines them into a dictionary in which each key maps to a list of all the appropriate values from the parameter dictionaries


def combine_dictionaries(list_of_dictionaries):
    combined_dictionaries = {}

    for individual_dictionary in list_of_dictionaries:
        for key_value in individual_dictionary:
            if key_value not in combined_dictionaries:
                combined_dictionaries[key_value] = []
            combined_dictionaries[key_value].append(individual_dictionary[key_value])

    return combined_dictionaries


# %%

# Calculate the average of all  combined dictionaries


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
# Function to create deep CNN

# This function takes as an input a list of dictionaries. Each element in the list is a new hidden layer in the model. For each
# layer the dictionary defines the layer to be used.


def create_models(model_shape, input_layer_dim):
    # parameter rate for l2 regularization
    regConst = 0.02

    # defining a stochastic gradient boosting optimizer
    sgd = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True, clipnorm=1.0)

    # define categorical_crossentrophy as the loss function (multi-class problem i.e. 3 age classes)

    cce = "categorical_crossentropy"
    # bce = 'binary_crossentropy'

    # input shape vector

    # change the input shape to avoid learning feautures independently. By changing the input shape to
    # (input_layer_dim, ) it will learn some combination of feautures with the learnable weights of the
    # network

    input_vec = tf.keras.Input(name="input", shape=(input_layer_dim,))

    for i, layerwidth in zip(range(len(model_shape)), model_shape):
        if i == 0:
            if model_shape[i]["type"] == "c":
                # Convolution1D layer, which will learn filters from spectra
                # signals with maxpooling1D and batch normalization:

                xd = tf.keras.layers.Conv1D(
                    name=("Conv" + str(i + 1)),
                    filters=model_shape[i]["filter"],
                    kernel_size=model_shape[i]["kernel"],
                    strides=model_shape[i]["stride"],
                    activation="relu",
                    kernel_regularizer=regularizers.l2(regConst),
                    kernel_initializer="he_normal",
                )(input_vec)
                xd = tf.keras.layers.BatchNormalization(
                    name=("batchnorm_" + str(i + 1))
                )(xd)
                xd = tf.keras.layers.MaxPooling1D(
                    pool_size=(model_shape[i]["pooling"])
                )(xd)

                # A hidden layer

            elif model_shape[i]["type"] == "d":
                xd = tf.keras.layers.Dense(
                    name=("d" + str(i + 1)),
                    units=model_shape[i]["width"],
                    activation="relu",
                    kernel_regularizer=regularizers.l2(regConst),
                    kernel_initializer="he_normal",
                )(input_vec)
                xd = tf.keras.layers.BatchNormalization(
                    name=("batchnorm_" + str(i + 1))
                )(xd)
                xd = tf.keras.layers.Dropout(name=("dout" + str(i + 1)), rate=0.5)(xd)

        else:
            if model_shape[i]["type"] == "c":
                # convolutional1D layer

                xd = tf.keras.layers.Conv1D(
                    name=("Conv" + str(i + 1)),
                    filters=model_shape[i]["filter"],
                    kernel_size=model_shape[i]["kernel"],
                    strides=model_shape[i]["stride"],
                    activation="relu",
                    kernel_regularizer=regularizers.l2(regConst),
                    kernel_initializer="he_normal",
                )(xd)
                xd = tf.keras.layers.BatchNormalization(
                    name=("batchnorm_" + str(i + 1))
                )(xd)
                xd = tf.keras.layers.MaxPooling1D(
                    pool_size=(model_shape[i]["pooling"])
                )(xd)

            elif model_shape[i]["type"] == "d":
                if model_shape[i - 1]["type"] == "c":
                    xd = tf.keras.layers.Flatten()(xd)

                xd = tf.keras.layers.Dropout(name=("dout" + str(i + 1)), rate=0.5)(xd)
                xd = tf.keras.layers.Dense(
                    name=("d" + str(i + 1)),
                    units=model_shape[i]["width"],
                    activation="relu",
                    kernel_regularizer=regularizers.l2(regConst),
                    kernel_initializer="he_normal",
                )(xd)
                xd = tf.keras.layers.BatchNormalization(
                    name=("batchnorm_" + str(i + 1))
                )(xd)

    # Project the vector onto a 3 unit output layer, and squash it with a
    # sigmoid activation:
    # x_age_group will have decoded inputs

    x_age_group = tf.keras.layers.Dense(
        name="age_group",
        units=2,
        activation="softmax",
        #   activation = 'sigmoid',
        kernel_regularizer=regularizers.l2(regConst),
        kernel_initializer="he_normal",
    )(xd)

    outputs = []
    for i in ["x_age_group"]:
        outputs.append(locals()[i])
    model = Model(inputs=input_vec, outputs=outputs)

    model.compile(loss=cce, metrics=["accuracy"], optimizer=sgd)
    model.summary()
    return model


# %%

# Convert age from strings to numbers
Age = []

for row in train_data["Cat3"]:
    if row == "01D":
        Age.append(1)

    elif row == "02D":
        Age.append(2)

    elif row == "03D":
        Age.append(3)

    elif row == "04D":
        Age.append(4)

    elif row == "05D":
        Age.append(5)

    elif row == "06D":
        Age.append(6)

    elif row == "07D":
        Age.append(7)

    elif row == "08D":
        Age.append(8)

    elif row == "09D":
        Age.append(9)

    elif row == "10D":
        Age.append(10)

    elif row == "11D":
        Age.append(11)

    elif row == "12D":
        Age.append(12)

    elif row == "13D":
        Age.append(13)

    elif row == "14D":
        Age.append(14)

    elif row == "15D":
        Age.append(15)

    else:
        Age.append(16)

print(Age)

train_data["Age"] = Age

# drop the column with age as string

train_data = train_data.drop(["Cat3"], axis=1)
train_data.head(5)


# %%

# define X (matrix of features) and y (vector of labels)

X = np.asarray(train_data.iloc[:, :-1])  # select all columns except the first one
y = np.asarray(train_data["Age"])

print("shape of X : {}".format(X.shape))
print("shape of y : {}".format(y.shape))


# %%

# Scale data into unit variance

scl = StandardScaler()
scaler = scl.fit(X=X)
X_new = scaler.transform(X=X)

# Dimension reduction with PCA, as Mwanga et al., 2022
pca = decomposition.PCA(n_components=8)
age_pca = pca.fit_transform(X_new)

print("shape of reduced X : {}".format(age_pca.shape))
print(np.unique(y))

# %%
# Renaming the age group into two classes
# Oganises the data into a format of lists of data, classes, labels.

y_age_group = np.where((y <= 9), 0, 0)
y_age_group = np.where((y >= 10), 1, y_age_group)

y_age_groups_list = [[ages] for ages in y_age_group]
age_group = MultiLabelBinarizer().fit_transform(np.array(y_age_groups_list))
# print('age_group', age_group)
age_group_classes = ["1-9", "10-16"]

# Labels default - all classification
labels_default, classes_default, outputs_default = (
    [age_group],
    [age_group_classes],
    ["x_age_group"],
)


# %%

# Function to train the model

# This function will split the data into training and validation, and call the create models function.
# This fucntion returns the model and training history.


def train_models(model_to_test, save_path):
    model_shape = model_to_test["model_shape"][0]
    model_name = model_to_test["model_name"][0]
    input_layer_dim = model_to_test["input_layer_dim"][0]
    model_ver_num = model_to_test["model_ver_num"][0]
    fold = model_to_test["fold"][0]
    y_train = model_to_test["labels"][0]
    X_train = model_to_test["features"][0]
    classes = model_to_test["classes"][0]
    outputs = model_to_test["outputs"][0]
    compile_loss = model_to_test["compile_loss"][0]
    compile_metrics = model_to_test["compile_metrics"][0]

    model = create_models(model_shape, input_layer_dim)

    #   model.summary()

    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=256,
        verbose=1,
        epochs=8000,
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=400, verbose=1, mode="auto"
            ),
            CSVLogger(
                save_path + model_name + "_" + str(model_ver_num) + ".csv",
                append=True,
                separator=";",
            ),
        ],
    )

    model.save(
        (
            save_path
            + model_name
            + "_"
            + str(model_ver_num)
            + "_"
            + str(fold)
            + "_"
            + "Model.h5"
        )
    )
    graph_history(history, model_name, model_ver_num, fold, save_path)

    return model, history


# Main training and prediction section for the PCA data

# Functionality:
# Define the CNN to be built.
# Build a folder to output data into.
# Call the model training.
# Organize outputs and call visualization for plotting and graphing.


outdir = "../Results/MLP_pca"
build_folder(outdir, False)


# set model parameters
# model size when data dimension is reduced to 8 principle componets

# Options
# Convolutional Layer:

#     type = 'c'
#     filter = optional number of filters
#     kernel = optional size of the filters
#     stride = optional size of stride to take between filters
#     pooling = optional width of the max pooling

# dense layer:

#     type = 'd'
#     width = option width of the layer

model_size = [  # {'type':'c', 'filter':8, 'kernel':2, 'stride':1, 'pooling':1},
    #  {'type':'c', 'filter':8, 'kernel':2, 'stride':1, 'pooling':1},
    #  {'type':'c', 'filter':8, 'kernel':2, 'stride':1, 'pooling':1},
    {"type": "d", "width": 500},
    {"type": "d", "width": 500},
    {"type": "d", "width": 500},
    {"type": "d", "width": 500},
    {"type": "d", "width": 500},
    #  {'type':'d', 'width':500},
    {"type": "d", "width": 500},
]


# Name the model
model_name = "CNN"
label = labels_default

# Split data into 10 folds for training/testing
# Define cross-validation strategy

num_folds = 5
seed = 42
random_seed = np.random.randint(0, 81470)
kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

# Features
features = age_pca

histories = []
averaged_histories = []
fold = 1
train_model = True

# Name a folder for the outputs to go into

savedir = outdir + "\Training_Folder"
build_folder(savedir, True)
savedir = outdir + "\Training_Folder\l"

# start model training on standardized data

start_time = time()
save_predicted = []
save_true = []


for train_index, test_index in kf.split(features):
    # Split data into test and train

    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = list(map(lambda y: y[train_index], label)), list(
        map(lambda y: y[test_index], label)
    )

    # Further divide training dataset into train and validation dataset
    # with an 90:10 split

    validation_size = 0.1
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, *y_train, test_size=validation_size, random_state=seed
    )

    # expanding to one dimension, because the conv layer expcte to, 1
    X_train = X_train.reshape([X_train.shape[0], -1])
    X_val = X_val.reshape([X_val.shape[0], -1])
    X_test = X_test.reshape([X_test.shape[0], -1])

    # Check the sizes of all newly created datasets
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_val:", X_val.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_val:", y_val.shape)
    # print("Shape of y_test:", y_test.shape)

    input_layer_dim = len(age_pca[0])

    model_to_test = {
        "model_shape": [model_size],  # defines the hidden layers of the model
        "model_name": [model_name],
        "input_layer_dim": [input_layer_dim],  # size of input layer
        "model_ver_num": [0],
        "fold": [fold],  # kf.split number on
        "labels": [y_train],
        "features": [X_train],
        "classes": [classes_default],
        "outputs": [outputs_default],
        # "compile_loss": [{'age_group': 'categorical_crossentropy'}],
        "compile_loss": [{"age_group": "binary_crossentropy"}],
        "compile_metrics": [{"age_group": "accuracy"}],
    }

    # Call function to train all the models from the dictionary
    model, history = train_models(model_to_test, savedir)
    histories.append(history)

    print(X_test.shape)

    # predict the unseen dataset/new dataset
    y_predicted = model.predict(X_test)

    # change the dimension of y_test to array
    y_test = np.asarray(y_test)
    y_test = np.squeeze(y_test)  # remove any single dimension entries from the arrays

    print("y predicted shape", y_predicted.shape)
    print("y_test", y_test.shape)

    # save predicted and true value in each iteration for plotting averaged confusion matrix

    for pred, tru in zip(y_predicted, y_test):
        save_predicted.append(pred)
        save_true.append(tru)

    hist = history.history
    averaged_histories.append(hist)

    # Plotting confusion matrix for each fold/iteration

    visualizeDL(
        histories,
        savedir,
        model_name,
        str(fold),
        classes_default[0],
        outputs_default,
        y_predicted,
        y_test,
    )
    # log_data(X_test, 'test_index', fold, savedir)

    fold += 1

    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.

    tf.compat.v1.keras.backend.clear_session()

    # Delete the Keras model with these hyper-parameters from memory.
    del model

save_predicted = np.asarray(save_predicted)
save_true = np.asarray(save_true)
print("save predicted shape", save_predicted.shape)
print("save.true shape", save_true.shape)

# Plotting an averaged confusion matrix

visualizeDL(
    1,
    savedir,
    model_name,
    "Averaged",
    classes_default[0],
    outputs_default,
    save_predicted,
    save_true,
)

end_time = time()
print("Run time : {} s".format(end_time - start_time))
print("Run time : {} m".format((end_time - start_time) / 60))
print("Run time : {} h".format((end_time - start_time) / 3600))

# %%

# combine all dictionaries together

combn_dictionar = combine_dictionaries(averaged_histories)
with open(
    "../Results/MLP_pca/Training_Folder/combined_history_dictionaries.txt",
    "w",
) as outfile:
    json.dump(combn_dictionar, outfile)

# find the average of all dictionaries

combn_dictionar_average = find_mean_from_combined_dicts(combn_dictionar)

# Plot averaged histories
graph_history_averaged(combn_dictionar_average)

# %%
# Loading dataset for prediction

df_new = pd.read_csv("../Data/test_an_fun_df.csv")

print(df_new.head())

# Checking class distribution in the data
print(Counter(df_new["Cat3"]))

# drops columns of no interest
df_new = df_new.drop(["Unnamed: 0"], axis=1)
df_new.head(10)

# %%

# Rename age from strings to numbers
Age_2 = []

for row in df_new["Cat3"]:
    if row == "01D":
        Age_2.append(1)

    elif row == "02D":
        Age_2.append(2)

    elif row == "03D":
        Age_2.append(3)

    elif row == "04D":
        Age_2.append(4)

    elif row == "05D":
        Age_2.append(5)

    elif row == "06D":
        Age_2.append(6)

    elif row == "07D":
        Age_2.append(7)

    elif row == "08D":
        Age_2.append(8)

    elif row == "09D":
        Age_2.append(9)

    elif row == "10D":
        Age_2.append(10)

    elif row == "11D":
        Age_2.append(11)

    elif row == "12D":
        Age_2.append(12)

    elif row == "13D":
        Age_2.append(13)

    elif row == "14D":
        Age_2.append(14)

    elif row == "15D":
        Age_2.append(15)

    else:
        Age_2.append(16)

print(Age_2)

df_new["Age"] = Age_2

# drop the column with Chronological Age and keep the age structure
df_new = df_new.drop(["Cat3"], axis=1)
df_new.head(5)

# %%

# predicting new dataset with a model trained PCA transformed data
# define matrix of features and vector of labels

X_valid = np.asarray(df_new.iloc[:, :-1])
y_valid = np.asarray(df_new["Age"])

print("shape of X_valid : {}".format(X_valid.shape))
print("shape of y_valid : {}".format(y_valid.shape))

print(np.unique(y_valid))
print("shape of X : {}".format(X_valid.shape))

# Standardise features with standardscaler
X_valid_new = scaler.transform(X=X_valid)

# Use PCA to transform test data
age_valid = pca.transform(X_valid_new)
age_valid = age_valid.reshape([age_valid.shape[0], -1])
# print(age_valid)
print(age_valid.shape)


# %%
# rename the age group into two classes

y_age_group_val = np.where((y_valid <= 9), 0, 0)
y_age_group_val = np.where((y_valid >= 10), 1, 0)

y_age_groups_list_val = [[ages_val] for ages_val in y_age_group_val]
age_group_val = MultiLabelBinarizer().fit_transform(np.array(y_age_groups_list_val))
age_group_classes_val = ["1-9", "10-16"]

labels_default_val, classes_default_val = [age_group_val], [age_group_classes_val]

# %%

# load model trained

loaded_model = load_model(
    "../Results/MLP_pca/Training_Folder/lCNN_0_3_Model.h5"
)

# change the dimension of y_test to array
y_validation = np.asarray(labels_default_val)
y_validation = np.squeeze(
    y_validation
)  # remove any single dimension entries from the arrays

# generates output predictions based on the X_input passed

predictions = loaded_model.predict(age_valid)

# computes the loss based on the X_input you passed, along with any other metrics requested in the metrics param
# when model was compiled

score = loaded_model.evaluate(age_valid, y_validation, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Calculating precision, recall and f-1 scores metrics for the predicted samples

cr_pca = classification_report(
    np.argmax(y_validation, axis=-1), np.argmax(predictions, axis=-1)
)
print(cr_pca)

# save classification report to disk
cr = pd.read_fwf(io.StringIO(cr_pca), header=0)
cr = cr.iloc[0:]
cr.to_csv(
    "../Results/MLP_pca/Training_Folder/classification_report.csv"
)

# %%

# Plot the confusion matrix for predcited samples
visualizeDL(
    2,
    savedir,
    model_name,
    "Test_set",
    classes_default_val[0],
    outputs_default,
    predictions,
    y_validation,
)

# %%
