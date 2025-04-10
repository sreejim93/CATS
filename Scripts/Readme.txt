## Clustering.py

This Python script implements dimensionality reduction techniques—UMAP, T-SNE, and PCA—to visualize high-dimensional aCGH (array CGH) data for breast cancer subtype classification.

The visualizations aim to reveal underlying patterns and relationships within the data, providing insights into how different breast cancer subtypes—HER2 positive, Hormone receptor positive (HR+), 
and Triple negative (TN)—are distributed across the samples.

The script allows users to:

Explore high-dimensional aCGH data: Reduce complex datasets of chromosomal DNA data to 2D or 3D visualizations for easier interpretation.

Identify clusters or subgroups of breast cancer subtypes: Visualize how samples belonging to different breast cancer subtypes (HER2+, HR+, and TN) relate to each other in a lower-dimensional space.

Gain insights into the molecular structure of the data: Uncover hidden structures or relationships between genetic alterations in aCGH data, which can inform classification models and treatment decisions.


## Evaluation.py
Contains a Python script for evaluating machine learning models on biological datasets. The script performs the following tasks:

Data Preprocessing: Loads and processes biological datasets, removing unnecessary columns and transposing the data for analysis.

Feature Selection: Utilizes SelectKBest with the f_classif scoring function to select the most relevant features from the dataset.

Model Evaluation: Trains and evaluates models (SVM, k-NN, and Logistic Regression) on the selected features. It includes:

Confusion Matrix: Visualizes and saves confusion matrix plots for model performance evaluation.

ROC Curve: Generates and saves ROC curves to assess the classification performance of models across multiple thresholds.

Model Accuracy: Calculates and displays the accuracy of each model on a test dataset.


