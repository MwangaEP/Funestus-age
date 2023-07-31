# %%
# This program uses XGBoost retrained with only important wavenumers to predict the age
# structure of Anopheles funestus mosquitoes collected from the wild

# import all libraries
import os
import io
import json
import ast
import itertools
import collections
from time import time
from tqdm import tqdm
import pickle

from itertools import cycle
import datetime

import numpy as np
import pandas as pd

from random import randint
from collections import Counter

from scipy import stats
from scipy.stats import randint

from sklearn.model_selection import (
    ShuffleSplit,
    train_test_split,
    StratifiedKFold,
    StratifiedShuffleSplit,
    KFold,
)
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    f1_score,
    recall_score,
    precision_score,
    precision_recall_fscore_support,
)

from plotting_utils import visualize

from imblearn.under_sampling import RandomUnderSampler

from sklearn import decomposition
from sklearn import manifold
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif

from sklearn.pipeline import Pipeline


from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

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


# %%

# Upload An. funestus train data for model training

train_data = pd.read_csv(
    "../Data/train_an_fun_df.csv"
)
print(train_data.head())

print(train_data.shape)

# Checking class distribution in the data
print(Counter(train_data["Cat3"]))

# drops columns of no interest
train_data = train_data.drop(["Unnamed: 0"], axis=1)
train_data.head(10)


# %%

# rename age from string to real numbers

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

# drop the column with age as a string and keep the age in intergers

train_data = train_data.drop(["Cat3"], axis=1)
train_data.head(5)

# %%

# Renaming the age group into two age classes

Age_group = []

for row in train_data["Age"]:
    if row <= 9:
        Age_group.append("1-9")

    else:
        Age_group.append("10-16")

print(Age_group)

train_data["Age_group"] = Age_group

# drop the column with Chronological Age and keep the age structure

train_data = train_data.drop(["Age"], axis=1)
train_data.head(5)

# %%

# define parameters

num_folds = 5  # split data into five folds
seed = 42  # seed value
random_seed = np.random.randint(0, 81470)
scoring = "accuracy"  # metric for model evaluation

# specify cross-validation strategy
kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

# %%

# select X_matrix and list of labels
X = train_data.iloc[:, :-1]
y = train_data["Age_group"]

print("shape of X : {}".format(X.shape))
print("shape of y : {}".format(y.shape))

with open(
    "C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\std_ML-fullwn\important_wavenumbers.txt"
) as json_file:
    important_wavenumb = json.load(json_file)

# Select the important wavenumbers from the main dataframe
train_features = X[
    [wavenumber for wavenumber in X.columns if wavenumber in important_wavenumb]
]
print("shape of X, reduced data: {}".format(train_features.shape))

# scale features
X = np.asarray(train_features)

scaler = StandardScaler().fit(X=X)
scaled_features = scaler.transform(X=X)

# %%

# train XGB classifier and tune its hyper-parameters with randomized grid search

# features = np.asarray(scaled_features)
labels = np.asarray(y)
print(np.unique(labels))

num_rounds = 5

classifier = XGBClassifier(objective="binary:logistic")

# set hyparameters

# Number of trees in XGB
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(3, 50, num=11)]
max_depth.append(None)
# Learning rate
learning_rate = [0.001, 0.01, 0.01, 0.1, 0.5]
subsample = [1.0, 0.9, 0.5, 0.1]
min_child_weight = [0.01, 0.05, 0.25, 0.75]
gamma = [0, 0.5, 2, 5, 10]
bytree = np.arange(0.1, 1.0, 0.01)
colsample_bylevel = np.round(np.arange(0.1, 1.0, 0.01))


param_grid = {
    "learning_rate": learning_rate,
    "max_depth": max_depth,
    "min_child_weight": min_child_weight,
    "gamma": gamma,
    "colsample_bytree": bytree,
    "n_estimators": n_estimators,
    "subsample": subsample,
    "colsample_bylevel": colsample_bylevel,
}

# prepare matrices of results

kf_results = pd.DataFrame()  # model parameters and global accuracy score
kf_per_class_results = []  # per class accuracy scores
# Evaluation_result = []
save_predicted = []  # save predicted values for plotting averaged confusion matrix
save_true = []  # save true values for plotting averaged confusion matrix

start = time()

for round in range(num_rounds):
    SEED = np.random.randint(0, 81470)

    for train_index, test_index in kf.split(scaled_features, labels):
        # Split data into test and train

        X_train, X_test = scaled_features[train_index], scaled_features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # split validation

        # validation_size = 0.05
        # X_train, X_val, y_train, y_val = train_test_split(X_train,
        #                                  y_train, test_size = validation_size, random_state = seed)

        # Check the sizes of all newly created datasets
        print("Shape of X_train:", X_train.shape)
        # print("Shape of X_val:", X_val.shape)
        print("Shape of X_test:", X_test.shape)
        print("Shape of y_train:", y_train.shape)
        # print("Shape of y_val:", y_val.shape)
        print("Shape of y_test:", y_test.shape)

        # generate models using all combinations of settings

        # RANDOMSED GRID SEARCH
        n_iter_search = 10
        rsCV = RandomizedSearchCV(
            verbose=1,
            estimator=classifier,
            param_distributions=param_grid,
            n_iter=n_iter_search,
            scoring=scoring,
            cv=kf,
            refit=True,
            n_jobs=-1,
        )

        rsCV_result = rsCV.fit(X_train, y_train)

        # print out results and give hyperparameter settings for best one
        means = rsCV_result.cv_results_["mean_test_score"]
        stds = rsCV_result.cv_results_["std_test_score"]
        params = rsCV_result.cv_results_["params"]
        for mean, stdev, param in zip(means, stds, params):
            print("%.2f (%.2f) with: %r" % (mean, stdev, param))

        # print best parameter settings
        print(
            "Best: %.2f using %s" % (rsCV_result.best_score_, rsCV_result.best_params_)
        )

        # Insert the best parameters identified by randomized grid search into the base classifier
        classifier = XGBClassifier(**rsCV_result.best_params_)

        # Fitting the best classifier
        # define the evaluation set
        # eval_set = [(X_train, y_train), (X_val, y_val)]
        classifier.fit(X_train, y_train)

        # ev_result = classifier.evals_result()
        # Evaluation_result.append(ev_result)

        # Predict X_test
        y_pred = classifier.predict(X_test)

        # Summarize outputs for plotting averaged confusion matrix

        for predicted, true in zip(y_pred, y_test):
            save_predicted.append(predicted)
            save_true.append(true)

        # summarize for plotting per class distribution

        classes = ["1-9", "10-16"]
        local_cm = confusion_matrix(y_test, y_pred, labels=classes)
        local_report = classification_report(y_test, y_pred, labels=classes)

        local_kf_results = pd.DataFrame(
            [
                ("Accuracy", accuracy_score(y_test, y_pred)),
                # ("params",str(rsCV_result.best_params_)),
                ("TRAIN", str(train_index)),
                ("TEST", str(test_index)),
                ("CM", local_cm),
                ("Classification report", local_report),
            ]
        ).T

        local_kf_results.columns = local_kf_results.iloc[0]
        local_kf_results = local_kf_results[1:]
        kf_results = kf_results.append(local_kf_results)

        # per class accuracy
        local_support = precision_recall_fscore_support(y_test, y_pred, labels=classes)[
            3
        ]
        local_acc = np.diag(local_cm) / local_support
        kf_per_class_results.append(local_acc)

elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(elapsed / 60, elapsed))


# %%

# plot averaged confusionfor the validation set

figure_name = "validation-02%"
classes = np.unique(np.sort(y))
visualize(figure_name, classes, save_predicted, save_true)

# %%

# #
# epochs = len(ev_result['validation_0']['error'])
# x_axis = range(0, epochs)

# sns.set(context = "paper",
#         style = "whitegrid",
#         palette = "deep",
#         font_scale = 2.0,
#         color_codes = True,
#         rc = ({'font.family': 'Helvetica'}))

# plt.figure(figsize = (6, 4))
# fig, ax = plt.subplots()
# ax.plot(x_axis, ev_result['validation_0']['error'], label='Train')
# ax.plot(x_axis, ev_result['validation_1']['error'], label='Test')
# ax.legend()
# plt.xlabel('Epochs', weight = 'bold')
# plt.ylabel('Classification Error rate', weight = 'bold')
# # plt.title('XGBoost learning curve', weight = 'bold')
# plt.savefig(('C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\std_ML\XGB_learning_curve.png'), dpi = 500, bbox_inches = 'tight')
# plt.show()


# %%

# Results

kf_results.to_csv(
    "C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\std_ML\crf_kfCV_record.csv",
    index=False,
)
kf_results = pd.read_csv(
    "C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\std_ML\crf_kfCV_record.csv"
)

# Accuracy distribution
crf_acc_distrib = kf_results["Accuracy"]
crf_acc_distrib.columns = ["Accuracy"]
crf_acc_distrib.to_csv(
    "C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\std_ML\crf_acc_distrib.csv",
    header=True,
    index=False,
)
crf_acc_distrib = pd.read_csv(
    "C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\std_ML\crf_acc_distrib.csv"
)
crf_acc_distrib = np.round(crf_acc_distrib, 2)
print(crf_acc_distrib)

# %%
# preparing dataframe for plotting per class accuracy

classes = ["1-9", "10-16"]
rf_per_class_acc_distrib = pd.DataFrame(kf_per_class_results, columns=classes)
rf_per_class_acc_distrib.dropna().to_csv(
    "C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\std_ML\_rf_per_class_acc_distrib.csv"
)
rf_per_class_acc_distrib = pd.read_csv(
    "C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\std_ML\_rf_per_class_acc_distrib.csv",
    index_col=0,
)
rf_per_class_acc_distrib = np.round(rf_per_class_acc_distrib, 1)
rf_per_class_acc_distrib_describe = rf_per_class_acc_distrib.describe()
rf_per_class_acc_distrib_describe.to_csv(
    "C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\std_ML\_rf_per_class_acc_distrib.csv"
)

# %%
# plotting class distribution
sns.set(
    context="paper",
    style="whitegrid",
    palette="deep",
    font_scale=2.0,
    color_codes=True,
    rc=({"font.family": "Helvetica"}),
)

plt.figure(figsize=(6, 4))

rf_per_class_acc_distrib = pd.melt(rf_per_class_acc_distrib, var_name="Label new")
sns.violinplot(x="Label new", y="value", data=rf_per_class_acc_distrib)
sns.despine(left=True)

plt.xticks(ha="right")
plt.yticks()
plt.ylim(ymin=0.2, ymax=1.0)
plt.xlabel(" ")
# plt.legend(' ', frameon = False)
plt.ylabel("Prediction accuracy", weight="bold")
plt.grid(False)
plt.tight_layout()
# plt.show()
plt.savefig(
    "C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\std_ML\_rf_per_class_acc_distrib.png",
    dpi=500,
    bbox_inches="tight",
)

# %%

# save the trained model to disk for future use

with open(
    "C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\std_ML\classifier.pkl", "wb"
) as fid:
    pickle.dump(classifier, fid)


# %%
# start by loading the new test data

df_new = pd.read_csv(
    "C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\set_to_test_an_fun_new.csv"
)
print(df_new.head())

# Checking class distribution in the data
print(Counter(df_new["Cat3"]))

# drops columns of no interest
df_new = df_new.drop(["Unnamed: 0"], axis=1)
df_new.head(10)

# %%
# rename age from string to numbers
Age = []

for row in df_new["Cat3"]:
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

df_new["Age"] = Age

# drop the column with Chronological Age and keep the age structure
df_new = df_new.drop(["Cat3"], axis=1)
df_new.head(5)


# %%

# Renaming the age group into three classes
Age_group = []

for row in df_new["Age"]:
    if row <= 9:
        Age_group.append("1-9")
    else:
        Age_group.append("10-16")

print(Age_group)

df_new["Age_group"] = Age_group

# drop the column with Chronological Age and keep the age structure
df_new = df_new.drop(["Age"], axis=1)
df_new.head(5)

# %%
# select X matrix of features and y list of labels

X_valid = df_new.iloc[:, :-1]
y_valid = df_new["Age_group"]

print("shape of X_valid : {}".format(X_valid.shape))
print("shape of y_valid : {}".format(y_valid.shape))

# Select the important wavenumbers from the main dataframe
new_data_val = X_valid[[x for x in X_valid.columns if x in important_wavenumb]]

print("shape of X : {}".format(new_data_val.shape))

y_valid = np.asarray(y_valid)
print(np.unique(y_valid))

# scale X test
X_valid = scaler.transform(X=np.asarray(new_data_val))


# %%
# loading the classifier from the disk
with open(
    "C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\std_ML-selected wavenumbers\classifier.pkl",
    "rb",
) as fid:
    classifier_loaded = pickle.load(fid)

# generates output predictions based on the X_input passed

predictions = classifier_loaded.predict(X_valid)

# Examine the accuracy of the model in predicting glasgow data

accuracy = accuracy_score(y_valid, predictions)
print("Accuracy:%.2f%%" % (accuracy * 100.0))

# compute precision, recall and f-score metrics

classes = ["1-9", "10-16"]
cr_report = classification_report(y_valid, predictions, labels=classes)
print(cr_report)

# %%

# save classification report to disk as a csv

cr = pd.read_fwf(io.StringIO(cr_report), header=0)
cr = cr.iloc[0:]
cr.to_csv(
    "C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\std_ML-selected wavenumbers\classification_report_.csv"
)

# %%

# plot the confusion matrix for the test data (glasgow data)
figure_name = "test_02"
classes = np.unique(np.sort(y_valid))
visualize(figure_name, classes, predictions, y_valid)

# %%