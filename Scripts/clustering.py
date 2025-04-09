#!/usr/bin/env python3
"""
This is a script that performs three types of cluster visualisation techiniques. They are UMAP, T-SNE and PCA.


Author: Lotte Bottema
Course: Bioinformatics for Translational Medicine
Project: CATS
Date: 17-05-2022
"""

import matplotlib.pyplot as plt
import umap
import os
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors



def make_umap(data, labels, random_state=42):

    # Scale the data and fit and transform it.
    data = StandardScaler().fit_transform(data)
    labels = labels.to_numpy()

    # Apply the UMAP
    umap_trans = umap.UMAP(random_state)
    umap_result = umap_trans.fit_transform(data)
    umap_result_df = pd.DataFrame({'umap_1': umap_result[:,0], 'umap_2': umap_result[:,1], 'label': labels.flatten()})

    # Plot the data
    _, ax = plt.subplots(1, figsize = (8,8))
    sns.scatterplot(x='umap_1', y='umap_2', hue='label', data=umap_result_df, ax=ax,s=120, palette="hls")
    ax.set_aspect('equal')
    ax.legend()#bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    ax.grid()
    plt.show()



def make_tsne(data, labels):
    # Scale the data and fit and transform it.
    data = StandardScaler().fit_transform(data)
    labels = labels.to_numpy()

    # Apply T-SNE
    tsne = TSNE(n_components=2)
    tsne_result = tsne.fit_transform(data)
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': labels.flatten()})

    # Plot the data.
    _, ax = plt.subplots(1, figsize = (8,8))
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120, palette="hls")
    ax.set_aspect('equal')
    ax.legend()#bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    ax.grid()
    plt.show()

def make_3dpca(data, labels):
    # Scale the data and fit and transform it.
    data = StandardScaler().fit_transform(data)

    # Apply the PCA
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(data)
    principalDf3 = pd.DataFrame(data = pcs, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
    finalDf3 = pd.concat([principalDf3, labels], axis = 1)

    # Plot the data after PCA
    fig = plt.figure(figsize = (8,8))
    ax = plt.axes(projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    # ax.set_zlabel('Principal Component 3', fontsize = 15)
    ax.set_title('3 component PCA', fontsize = 20)

    targets = ['HER2+', 'HR+', 'Triple Neg']
    colors = ['indianred', 'lightgreen', 'royalblue']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf3['Subgroup'] == target
        ax.scatter(finalDf3.loc[indicesToKeep, 'principal component 1']
                , finalDf3.loc[indicesToKeep, 'principal component 2']
                , finalDf3.loc[indicesToKeep, 'principal component 3']
                , c = color)
    ax.legend(targets)
    ax.grid()
    plt.show()

def make_2dpca(data, labels):

    # Scale the data and fit and transform it.
    data = StandardScaler().fit_transform(data)

    # Apply the PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(data)
    principalDf = pd.DataFrame(data = pcs, columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, labels], axis = 1)


    # Plot the data after PCA
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)

    targets = ['HER2+', 'HR+', 'Triple Neg']
    colors = ['indianred', 'lightgreen', 'royalblue']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['Subgroup'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()


if __name__ == '__main__':
    # Obtain the data and labels in dataframes
    data =  pd.read_csv(os.path.join("..", "output", "best_features.csv"), sep=";")
    labels = pd.read_csv(os.path.join("..", "data", "Train_clinical.csv"), sep=";")

    # Drop the sample columns in both labels and data
    labels = labels.drop(columns=['Sample'])

    # Apply the visualisation techniques
    make_3dpca(data, labels)
    make_tsne(data, labels)
    make_umap(data, labels)