#!/usr/bin/env python3

"""
This is a script does a scoring evaluation of the models with correct hyper-parameters and selected features

Author: Aron van Beelen
Course: Bioinformatics for Translational Medicine
Project: CATS
Date: 10-05-2022
"""


import os
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from yellowbrick.classifier import ROCAUC


def load_data(dataset, labels):
    """
       This function reads the input file and makes it into a DataFrame
    """
    # load dataset as a DataFrame
    dataset = pd.read_csv(dataset, sep="\t")

    # load labels as a DataFrame
    labels = pd.read_csv(labels, sep="\t")

    # remove chromosome, Start, End and Nclone columns to make a "simple" data set
    dataset.drop(columns=['Chromosome', 'Start', 'End', 'Nclone'], inplace=True)

    # transpose DataFrame such that the probes (samples) are rows
    dataset = dataset.transpose()
    dataset = dataset.reset_index(level=0)
    dataset = dataset.rename(columns={"index": "Sample"})

    # merge target labels to dataset
    dataset = pd.merge(dataset, labels, on='Sample')

    # remove "Sample" column
    dataset = dataset.drop(columns=["Sample"], axis=1)

    # split into input (X) and output (y) variables
    X = dataset.values[:, :-1] + 1
    y = dataset.values[:, -1]

    return X, y


def create_confusion_matrix(model, model_name, predictions, y_test):
    """
    Create confusion matrices for a given model.

    link: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
    """
    # perform confusion matrix calculations
    cm = confusion_matrix(y_test, predictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()

    # plot attributes
    plt.suptitle(f"Confusion matrix of '{model_name}' model")
    plt.grid(False)

    # save the confusion matrix
    plt.savefig(os.path.join("..", "plots", f"confusion_matrix_{model_name}.png"))

    # verbose
    print(f"Saved confusion matrix plot for {model_name}!")

    plt.clf()
    return


def create_ROC_curve(model, X_train, y_train, X_test, y_test):
    """
    Create multiclass ROC curve

    link: https://medium.com/swlh/how-to-create-an-auc-roc-plot-for-a-multiclass-model-9e13838dd3de
    """

    # Creating visualization with the readable labels
    visualizer = ROCAUC(model)

    # Fitting to the training data first then scoring with the test data
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)

    # save the confusion matrix
    visualizer.finalize()
    plt.savefig(os.path.join("..", "plots", f"ROC_{model_name}.png"))

    # verbose
    print(f"Saved ROC curve for {model_name}!")

    return visualizer


if __name__ == '__main__':
    dataset = os.path.join("..", "data", "Train_call.txt")
    labels = os.path.join("..", "data", "Train_clinical.txt")

    # loop over every model
    for model, n_of_features, model_name in zip(
            [
                SVC(C=2.100000000000001, gamma="auto", kernel="rbf"),
                KNeighborsClassifier(leaf_size=1, n_neighbors=1, p=2),
                LogisticRegression(C=0.01, penalty='l2', solver='saga')
            ],
            [604 , 604, 604],
            ["SVM", "k-NN", "Logistic Regression"]
        ):

        # load data
        X, y = load_data(dataset, labels)

        # perform feature selection using f_classif
        fs = SelectKBest(score_func=f_classif, k=n_of_features)
        fs.fit(X, y)
        X_trans = fs.transform(X)

        # perform train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.33, random_state=1, stratify=y)

        # perform model fit
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # create and save confusion matrices
        create_confusion_matrix(model, model_name, predictions, y_test)

        # create and save ROC curve for multiclass probles
        create_ROC_curve(model, X_train, y_train, X_test, y_test)

        # calculate the accuracy
        acc = accuracy_score(y_test, predictions)

        input(f"{model_name} = {acc}")
