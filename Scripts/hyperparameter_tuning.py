#!/usr/bin/env python3
"""
This is a script that performs hyper-parameter tuning for SVM, Logistic Regression, and k-NN.

Author: Aron van Beelen & Sreejita Mazumder
Course: Bioinformatics for Translational Medicine
Project: CATS
Date: 10-05-2022
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import f_classif
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
from feature_selection import load_data


def hyperparameter_optimization(X, model, space, model_name):
    """
    This function performs feature selection and hyperparameter tuning in parallel.
    It makes a plot with the amount of features, corresponding accuracies, and hyper-parameters

    """
    # initialise a matplotlib figure
    plt.figure()

    # initialise the outer loop with 7 splits
    cv_outer = KFold(n_splits=7, shuffle=True, random_state=1)

    # hyper-parameter dictionary
    hyperparameters = {}

    # create two dictionaries that will contain accuracy scores
    cv_acc_dict = {}
    acc_dict = {}

    # create a list containing number of features used
    step = 20
    n_features = [1] + list(range(step, len(X.T) + 1, step)) + [len(X.T)]

    # start cross-validating procedure
    # split the dataset according to the amount of folds in the outer loop, which is 7
    for train_ix, test_ix in cv_outer.split(X):
        # perform feature selection using f_classif
        fs = SelectKBest(score_func=f_classif, k=604)
        fs.fit(X, y)
        X_trans = fs.transform(X)

        # split the dataset to input (X) and response (y) train-, test set
        X_train, X_test = X_trans[train_ix, :], X_trans[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        # initialise the inner loop with 3 splits
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)

        # initialise the Grid search such that the hyper-parameters can be find within the inner loop
        if model_name == "Logistic Regression":
            for grid in space:
                search = GridSearchCV(model, grid, scoring='accuracy', cv=cv_inner, refit=True)

                # execute Grid search
                result = search.fit(X_train, y_train)

                # get the model with the hyper-parameters
                best_model = result.best_estimator_

                # evaluate model on the hold out dataset
                yhat = best_model.predict(X_test)

                # evaluate the model
                acc = accuracy_score(y_test, yhat)

                # verbose progress
                print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))

                hyperparameters[float(result.best_score_) + float(acc)] = result.best_params_
        else:
            search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)

            # execute Grid search
            result = search.fit(X_train, y_train)

            # get the model with the hyper-parameters
            best_model = result.best_estimator_

            # evaluate model on the hold out dataset
            yhat = best_model.predict(X_test)

            # evaluate the model
            acc = accuracy_score(y_test, yhat)

            # verbose progress
            print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))

            hyperparameters[float(result.best_score_) + float(acc)] = result.best_params_

    return (max(hyperparameters.keys())/2), hyperparameters[max(hyperparameters.keys())]


if __name__ == '__main__':
    dataset = os.path.join("..", "data", "Train_call.txt")
    labels = os.path.join("..", "data", "Train_clinical.txt")

    # load data
    X, y = load_data(dataset, labels)

    # make grid spaces for hyper-parameter tuning
    space_SVM = {"gamma": ["scale", "auto"],
                 "C": list(np.arange(1, 10.0, 0.1)),
                 "kernel": ['rbf', 'poly', 'sigmoid']
                 }
    space_knn = {"leaf_size": list(range(1, 25)),
                 "n_neighbors": list(range(1, 5)),
                 "p": [1, 2],
                 }
    space_lr = [{'solver': ["lbfgs"],
                "penalty": ["none", "l2"],
                "C": [100, 10, 1.0, 0.1, 0.01],
                },
                {'solver': ["newton-cg"],
                 "penalty": ["none", "l2"],
                 "C": [100, 10, 1.0, 0.1, 0.01],
                 },
                {'solver': ["liblinear"],
                 "penalty": ["none", "l2"],
                 "C": [100, 10, 1.0, 0.1, 0.01],
                 },
                {'solver': ["newton-cg"],
                 "penalty": ["l1", "l2"],
                 "C": [100, 10, 1.0, 0.1, 0.01],
                 },
                {'solver': ["sag"],
                 "penalty": ["none", "l2"],
                 "C": [100, 10, 1.0, 0.1, 0.01],
                 },
                {'solver': ["saga"],
                 "penalty": ["none", "l2", "l1", "elasticnet"],
                 "C": [100, 10, 1.0, 0.1, 0.01],
                 },
                ]

    # execute hyper-parameter tuning
    with open(os.path.join("..", "output", "hyperparameters.txt"), "w") as file:
        for model, space, model_name in zip(
                [svm.SVC(random_state=1), KNeighborsClassifier(), LogisticRegression(random_state=1)],
                [space_SVM, space_knn, space_lr],
                ["SVM", "kNN", "Logistic Regression"]):
            acc, parameters = hyperparameter_optimization(X, model, space, model_name)
            print(f"DONE> Model {model_name} (mean acc={acc}): {parameters}")
            file.write(f"Model {model_name} (mean acc={acc}): {parameters}\n")
