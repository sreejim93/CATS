#!/usr/bin/env python3
"""
This is a script that performs feature selection using clinical data form CATS project

Author: Aron van Beelen & Lotte Bottema
Course: Bioinformatics for Translational Medicine
Project: CATS
Date: 10-05-2022
"""

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import pandas as pd
import os

from sqlalchemy import true

# load dataset
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


def determine_best_features(X,y):
    fs = SelectKBest(score_func=f_classif, k='all')
    fs.fit(X, y)
    high_fs = [i for i in fs.pvalues_ if i < (0.05/3)]
    print(f"The amount of features with a p-value smaller than 0.05 are: {len(high_fs)}")

# feature selection
def select_features(X, y, num_feat=365):
    # select all features and score relevancy to its class label
    fs = SelectKBest(score_func=f_classif, k=num_feat)
    # fit the data and transform it
    X = fs.fit_transform(X,y)
    feature_indices = fs.get_support(indices=true)
    best_feature_cols = [i for i in feature_indices]
    print(best_feature_cols)
    return X, y, fs, best_feature_cols

if __name__ == '__main__':
    dataset = os.path.join("..", "data", "Train_call.txt")
    labels = os.path.join("..", "data", "Train_clinical.txt")
    arrays = pd.read_csv(labels, sep="\t")
    arrays = arrays.values[:,0]

    # load data
    X, y = load_data(dataset, labels)

    # determine the amount of features to use (research purpose only)
    determine_best_features(X, y)
    


    # feature selection 
    X, y, fs, _ = select_features(X, y)

    # create DataFrame with selected features
    data = pd.DataFrame(data=fs.scores_, index=[x for x in range(len(fs.scores_))], columns=["score"])

    # sort the DataFrame by score
    data = data.sort_values(by=['score'], ascending=False)

    # open dataset, merge scores to it based on indices
    dataset = pd.read_csv(dataset, sep="\t")
    features = dataset.iloc[data.index]
    features = features.join(data)

    best_features = pd.DataFrame(X)

    # write the selected features to a file
    best_features.to_csv(os.path.join("..", "output","best_features.csv"), sep=";", header=True)
    features.to_csv(os.path.join("..", "output","features.csv"), sep=";", header=True)
    print("Done")