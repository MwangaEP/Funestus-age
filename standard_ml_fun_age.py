#%%
# This program uses standard machine learning to predict the age structure of 
# of Anopheles funestus mosquitoes collected from the wild

# Principal component analysis is used to reduce the dimensionality of the data

# import all libraries

import this
import os
import io
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

from sklearn.model_selection import ShuffleSplit, train_test_split, StratifiedKFold, StratifiedShuffleSplit, KFold 
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score, precision_recall_fscore_support

from imblearn.under_sampling import RandomUnderSampler

from sklearn import decomposition
from sklearn import manifold
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.pipeline import Pipeline


from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import BernoulliRBM
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from matplotlib import pyplot
import matplotlib.pyplot as plt # for making plots
import seaborn as sns
sns.set(context = "paper",
        style = "whitegrid",
        palette = "deep",
        font_scale = 2.0,
        color_codes = True,
        rc = ({'font.family': 'Helvetica'}))
# %matplotlib inline
plt.rcParams["figure.figsize"] = [6,4]

#%%

# This normalizes the confusion matrix and ensures neat plotting for all outputs.
# Function for plotting confusion matrcies

def plot_confusion_matrix(cm, classes,
                          normalize = True,
                          title = 'Confusion matrix',
                          xrotation=0,
                          yrotation=0,
                          cmap=plt.cm.Purples,
                          printout = False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if printout:
            print("Normalized confusion matrix")
    else:
        if printout:
            print('Confusion matrix')

    if printout:
        print(cm)
    
    plt.figure(figsize=(6,4))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    classes = classes
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=xrotation)
    plt.yticks(tick_marks, classes, rotation=yrotation)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', weight = 'bold')
    plt.xlabel('Predicted label', weight = 'bold')
    # plt.show()
    plt.savefig(("C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\std_ML\Confusion_Matrix_" + figure_name + "_" + ".png"), dpi = 500, bbox_inches="tight")
   

#%%
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
    classes = ['1 - 9', '10 - 16']
    cnf_matrix = confusion_matrix(classes_true, classes_pred, labels = classes)
    plot_confusion_matrix(cnf_matrix, classes)


#%%

# Loading dataset for prediction 
# Upload An. funestus train data for model training

train_data = pd.read_csv("C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\set_to_train_an_fun_new.csv")
print(train_data.head())

print(train_data.shape)

# Checking class distribution in the data
print(Counter(train_data["Cat3"]))

# drops columns of no interest
train_data = train_data.drop(['Unnamed: 0'], axis = 1)
train_data.head(10)


#%%
Age = []

for row in train_data['Cat3']:
    if row == '01D':
        Age.append(1)
    
    elif row == '02D':
        Age.append(2)
    
    elif row == '03D':
        Age.append(3)

    elif row == '04D':
        Age.append(4)

    elif row == '05D':
        Age.append(5)

    elif row == '06D':
        Age.append(6)

    elif row == '07D':
        Age.append(7)

    elif row == '08D':
        Age.append(8)

    elif row == '09D':
        Age.append(9)

    elif row == '10D':
        Age.append(10)

    elif row == '11D':
        Age.append(11)

    elif row == '12D':
        Age.append(12)

    elif row == '13D':
        Age.append(13)

    elif row == '14D':
        Age.append(14)

    elif row == '15D':
        Age.append(15)

    else:
        Age.append(16)

print(Age)

train_data['Age'] = Age

# drop the column with Chronological Age and keep the age structure
train_data = train_data.drop(['Cat3'], axis = 1) 
train_data.head(5)


#%%

# Renaming the age group into three classes

Age_group = []

for row in train_data['Age']:
    if row <= 5:
        Age_group.append('1 - 9')

    else:
        Age_group.append('10 - 16')

print(Age_group)

train_data['Age_group'] = Age_group

# drop the column with Chronological Age and keep the age structure
train_data = train_data.drop(['Age'], axis = 1) 
train_data.head(5)

#%%

# define parameters
num_folds = 5 # split data into five folds
seed = 42 # seed value
scoring = 'accuracy' # metric for model evaluation

# specify cross-validation strategy
kf = KFold(n_splits = num_folds, shuffle = True, random_state = seed)

# make a list of models to test
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('LR', LogisticRegressionCV(multi_class = 'ovr', cv = kf, max_iter = 500, random_state = seed)))
models.append(('SVM', SVC(kernel = 'linear', random_state = seed)))
models.append(('RF', RandomForestClassifier(n_estimators = 1000, random_state = seed)))
models.append(('XGBoost', XGBClassifier(random_state = seed, n_estimators = 1000)))
models.append(('DT', DecisionTreeClassifier(random_state = seed)))
models.append(('MLP', MLPClassifier(hidden_layer_sizes = 500, activation = 'logistic', 
                                    solver = 'sgd', alpha = 0.01, learning_rate_init = .01,
                                    max_iter = 3000, early_stopping = True)))
# models.append(('CatXGB', CatBoostClassifier(
#                            depth = 2,
#                            learning_rate = 0.01,
#                            loss_function = 'Logloss',
#                            n_estimators = 1000,
#                            verbose=False)))




#%%
X = train_data.iloc[:,:-1] # select all columns except the first one 
y = train_data["Age_group"]

print('shape of X : {}'.format(X.shape))
print('shape of y : {}'.format(y.shape))

# standardize inputs and transform them into lower dimension

# A pipeline containing standardization and PCA algorithm

pca_pipe = Pipeline([('scaler', StandardScaler()),
                      ('pca', decomposition.KernelPCA(n_components = 8, kernel = 'linear'))])

# Transform data into  principal componets 

#%%
age_pca = pca_pipe.fit_transform(X)
print('First five observation : {}'.format(age_pca[:5]))


#%%

X = np.asarray(age_pca)
y = np.asarray(y)
print(np.unique(y))


results = []
names = []

skf = StratifiedKFold(n_splits = num_folds, random_state=seed, shuffle=True)

for name, model in models:
    cv_results = cross_val_score(
        model, X, y, cv = kf, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = 'Cross validation score for {0}: {1:.2%}'.format(
        name, cv_results.mean(), cv_results.std()
    )
    print(msg)

#%%

# Plotting the algorithm selection 

sns.set(context = 'paper',
        style = 'whitegrid',
        palette = 'deep',
        font_scale = 2.0,
        color_codes = True,
        rc = ({'font.family': 'Dejavu Sans'}))

plt.figure(figsize = (6, 4))
sns.boxplot(x = names, y = results, width = .4)
sns.despine(offset = 10, trim = True)
plt.xticks(rotation = 90)
plt.yticks((0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
plt.ylim(ymin = 0.4, ymax = 1.0)
plt.ylabel('Accuracy', weight = 'bold')
plt.tight_layout()
# plt.show()
plt.savefig("C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\std_ML\_algorithm_sel_", dpi = 500, bbox_inches="tight")

#%%
# def learning_curve_XGB():
#     print()
#     print('**How we can visualise XGBoost model with learning curves**')


#     # # load data
#     # dataset = loadtxt('pima.indians.diabetes.data.csv', delimiter=",")

#     # # split data into X and y
#     # X = dataset[:,0:8]
#     # Y = dataset[:,8]

#     # split data into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=4)

#     # fit model no training data
#     model = XGBClassifier(random_state = seed, n_estimators = 1000)
#     eval_set = [(X_train, y_train), (X_test, y_test)]
#     model.fit(X_train, y_train, early_stopping_rounds=50, eval_metric=["merror", "mlogloss"], eval_set=eval_set, verbose=True)

#     # make predictions for test data
#     y_pred = model.predict(X_test)
#     # predictions = [round(value) for value in y_pred]

#     # evaluate predictions
#     accuracy = accuracy_score(y_test, y_pred)
#     print("Accuracy: %.2f%%" % (accuracy * 100.0))

#     # retrieve performance metrics
#     results = model.evals_result()
#     epochs = len(results['validation_0']['merror'])
#     x_axis = range(0, epochs)

#     # plot log loss
#     fig, ax = pyplot.subplots(figsize=(12,12))
#     ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
#     ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
#     ax.legend()
#     pyplot.ylabel('Log Loss')
#     pyplot.title('XGBoost Log Loss')
#     pyplot.show()

#     # plot classification error
#     fig, ax = pyplot.subplots(figsize=(12,12))
#     ax.plot(x_axis, results['validation_0']['merror'], label='Train')
#     ax.plot(x_axis, results['validation_1']['merror'], label='Test')
#     ax.legend()
#     pyplot.ylabel('Classification Error')
#     pyplot.title('XGBoost Classification Error')
#     pyplot.show()

# learning_curve_XGB()


#%%
# train XGB classifier and tune its hyper-parameters with randomized grid search 

print(np.unique(y))

num_rounds = 5
classifier = XGBClassifier(n_estimators = 100)

# classifier = RandomForestClassifier(n_estimators = 1000, random_state = seed)

# # classifier = MLPClassifier(hidden_layer_sizes = 500, activation = 'logistic', 
#                                     solver = 'sgd', alpha = 0.01, learning_rate_init = .1,
#                                     max_iter = 3000, early_stopping = True)


# set hyparameter

# estimators = [100, 500, 1000]
rate = [0.05, 0.10, 0.15, 0.20, 0.30]
depth = [2, 3, 4, 5, 6, 8, 10, 12, 15]
child_weight = [1, 3, 5, 7]
gamma = [0.0, 0.1, 0.2, 0.3, 0.4]
bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]

param_grid = dict(learning_rate = rate, max_depth = depth,
                min_child_weight = child_weight, gamma = gamma, colsample_bytree = bytree)


# prepare matrices of results
kf_results = pd.DataFrame() # model parameters and global accuracy score
kf_per_class_results = [] # per class accuracy scores

save_predicted = [] # save predicted values for plotting averaged confusion matrix
save_true = [] # save true values for plotting averaged confusion matrix

start = time()

for round in range(num_rounds):
    SEED = np.random.randint(0, 81470)
    
    for train_index, test_index in kf.split(X, y):

        # Split data into test and train

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # split validation
        # validation_size = 0.1
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
        rsCV = RandomizedSearchCV(verbose = 1,
                    estimator = classifier, param_distributions = param_grid, n_iter = n_iter_search, 
                                scoring = scoring, cv = kf)
        
        rsCV_result = rsCV.fit(X_train, y_train)

        # print out results and give hyperparameter settings for best one
        means = rsCV_result.cv_results_['mean_test_score']
        stds = rsCV_result.cv_results_['std_test_score']
        params = rsCV_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%.2f (%.2f) with: %r" % (mean, stdev, param))

        # print best parameter settings
        print("Best: %.2f using %s" % (rsCV_result.best_score_,
                                    rsCV_result.best_params_))

        # Insert the best parameters identified by randomized grid search into the base classifier
        classifier = XGBClassifier(nthread=1, seed=SEED, **rsCV_result.best_params_)

        # Fitting the best classifier
        # eval_set = [(X_val, y_val)]
        # classifier.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)

        classifier.fit(X_train, y_train)

        # Predict X_test
        y_pred = classifier.predict(X_test)

        # Summarize outputs for plotting averaged confusion matrix

        for predicted, true in zip(y_pred, y_test):
            save_predicted.append(predicted)
            save_true.append(true)

        # summarize for plotting per class distribution

        classes = ['1 - 9', '10 - 16']
        local_cm = confusion_matrix(y_test, y_pred, labels = classes)
        local_report = classification_report(y_test, y_pred, labels = classes)

        local_kf_results = pd.DataFrame([("Accuracy", accuracy_score(y_test, y_pred)),
                                        # ("params",str(rsCV_result.best_params_)),
                                        ("TRAIN",str(train_index)),
                                        ("TEST",str(test_index)),
                                        ("CM", local_cm),
                                        ("Classification report",
                                        local_report)]).T

        local_kf_results.columns = local_kf_results.iloc[0]
        local_kf_results = local_kf_results[1:]
        kf_results = kf_results.append(local_kf_results)

        # per class accuracy
        local_support = precision_recall_fscore_support(y_test, y_pred, labels = classes)[3]
        local_acc = np.diag(local_cm)/local_support
        kf_per_class_results.append(local_acc)

elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
    elapsed / 60, elapsed))


# %%

# plot confusion averaged for the validation set
figure_name = 'validation_5%'
classes = np.unique(np.sort(y))
visualize(figure_name, classes, save_true, save_predicted)

# %%
# preparing dataframe for plotting per class accuracy

classes = ['1 - 9', '10 - 16']
rf_per_class_acc_distrib = pd.DataFrame(kf_per_class_results, columns = classes)
rf_per_class_acc_distrib.dropna().to_csv("C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\std_ML\_rf_per_class_acc_distrib.csv")
rf_per_class_acc_distrib = pd.read_csv("C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\std_ML\_rf_per_class_acc_distrib.csv", index_col=0)
rf_per_class_acc_distrib = np.round(rf_per_class_acc_distrib, 1)
rf_per_class_acc_distrib_describe = rf_per_class_acc_distrib.describe()
rf_per_class_acc_distrib_describe.to_csv("C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\std_ML\_rf_per_class_acc_distrib.csv")

#%%
# plotting class distribution
sns.set(context = 'paper',
        style = 'whitegrid',
        palette = 'deep',
        font_scale = 2.0,
        color_codes = True,
        rc = ({'font.family': 'Helvetica'}))

plt.figure(figsize = (6, 4))

rf_per_class_acc_distrib = pd.melt(rf_per_class_acc_distrib, var_name="Label new")
sns.violinplot(x = "Label new", y = "value", cut = 0, data = rf_per_class_acc_distrib)
sns.despine(left=True)

plt.xticks(ha="right")
plt.yticks()
plt.ylim(ymin = 0.5, ymax = 1.0)
plt.xlabel(" ")
# plt.legend(' ', frameon = False)
plt.ylabel("Prediction accuracy\n ({0:.2f} ± {1:.2f})".format(rf_per_class_acc_distrib["value"].mean(),rf_per_class_acc_distrib["value"].sem()), weight="bold")
plt.grid(False)
plt.tight_layout()
# plt.show()
plt.savefig("C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\std_ML\_rf_per_class_acc_distrib.png", dpi = 500, bbox_inches="tight")

# %%

# save the trained model to disk for future use

with open('C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\std_ML\classifier.pkl', 'wb') as fid:
     pickle.dump(classifier, fid)


# %%
# Loading new dataset for prediction (Glasgow dataset)
# start by loading the new test data 

df_new = pd.read_csv("C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\set_to_test_an_fun_new.csv")
print(df_new.head())

# Checking class distribution in the data
print(Counter(df_new["Cat3"]))

# drops columns of no interest
df_new = df_new.drop(['Unnamed: 0'], axis=1)
df_new.head(10)

# %%
Age = []

for row in df_new['Cat3']:
    if row == '01D':
        Age.append(1)
    
    elif row == '02D':
        Age.append(2)
    
    elif row == '03D':
        Age.append(3)

    elif row == '04D':
        Age.append(4)

    elif row == '05D':
        Age.append(5)

    elif row == '06D':
        Age.append(6)

    elif row == '07D':
        Age.append(7)

    elif row == '08D':
        Age.append(8)

    elif row == '09D':
        Age.append(9)

    elif row == '10D':
        Age.append(10)

    elif row == '11D':
        Age.append(11)

    elif row == '12D':
        Age.append(12)

    elif row == '13D':
        Age.append(13)

    elif row == '14D':
        Age.append(14)

    elif row == '15D':
        Age.append(15)

    else:
        Age.append(16)

print(Age)

df_new['Age'] = Age

# drop the column with Chronological Age and keep the age structure
df_new = df_new.drop(['Cat3'], axis = 1) 
df_new.head(5)


#%%

# Renaming the age group into three classes
Age_group = []

for row in df_new['Age']:
    if row <= 5:
        Age_group.append('1 - 9')
    else:
        Age_group.append('10 - 16')

print(Age_group)

df_new['Age_group'] = Age_group

# drop the column with Chronological Age and keep the age structure
df_new = df_new.drop(['Age'], axis = 1) 
df_new.head(5)

#%%
# select X matrix of features and y list of labels

X_valid = df_new.iloc[:,:-1]
y_valid = df_new["Age_group"]

print('shape of X_valid : {}'.format(X_valid.shape))
print('shape of y_valid : {}'.format(y_valid.shape))

y_valid = np.asarray(y_valid)
print(np.unique(y_valid))

# tranform matrix of features with PCA 

age_valid = pca_pipe.fit_transform(X_valid)
print('First five observation : {}'.format(age_valid[:5]))
# transform age_valid as arrays
age_valid = np.asarray(age_valid)

#%%
# loading the classifier from the disk
with open('C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\std_ML\classifier.pkl', 'rb') as fid:
     classifier_loaded = pickle.load(fid)

# generates output predictions based on the X_input passed

predictions = classifier_loaded.predict(age_valid)

# Examine the accuracy of the model in predicting glasgow data 

accuracy = accuracy_score(y_valid, predictions)
print("Accuracy:%.2f%%" %(accuracy * 100.0))

# compute precision, recall and f-score metrics

classes = ['1 - 9', '10 - 16']
cr_pca = classification_report(y_valid, predictions, labels = classes)
print(cr_pca)

#%%

# save classification report to disk as a csv

cr = pd.read_fwf(io.StringIO(cr_pca), header=0)
cr = cr.iloc[1:]
cr.to_csv('C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\std_ML\classification_report_.csv')

#%%

# plot the confusion matrix for the test data (glasgow data)
figure_name = 'test_5'
classes = np.unique(np.sort(y_valid))
visualize(figure_name, classes, predictions, y_valid)

