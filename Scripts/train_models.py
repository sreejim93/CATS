#!/usr/bin/env python3
"""
This is a script that can be used to train the three models (knn, svc and logistic regression) using the hyperparameters obtained from the hyperparameter tuning.

Author: Lotte Bottema
Course: Bioinformatics for Translational Medicine
Project: CATS
Date: 20-05-2022
"""


import pandas as pd
import os
import pickle

from sqlalchemy import false
from feature_selection import load_data
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Removes columns which were not selected by SelectKBest(), base on the columns index.
def get_best_features(data):
    data = pd.DataFrame(data)
    feature_cols = [299, 301, 302, 303, 304, 305, 306, 307, 308, 309, 311, 312, 354, 461, 462, 463, 464, 465, 474, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 610, 611, 657, 663, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 744, 745, 746, 757, 758, 759, 769, 771, 772, 773, 814, 817, 818, 819, 820, 822, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 868, 870, 873, 874, 888, 998, 999, 1000, 1001, 1002, 1003, 1004, 1006, 1015, 1016, 1022, 1024, 1025, 1026, 1027, 1034, 1035, 1050, 1091, 1302, 1306, 1559, 1562, 1563, 1564, 1565, 1566, 1567, 1569, 1572, 1573, 1580, 1581, 1582, 1584, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679, 1680, 1681, 1682, 1683, 1684, 1685, 1686, 1687, 1688, 1690, 1691, 1693, 1878, 1879, 1881, 1895, 1897, 1899, 1900, 1902, 1903, 1906, 1907, 1908, 1909, 1910, 1949, 1951, 1952, 1959, 1960, 1966, 1971, 1972, 1973, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2026, 2027, 2028, 2029, 2032, 2034, 2035, 2036, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2048, 2049, 2050, 2051, 2052, 2054, 2055, 2056, 2057, 2058, 2059, 2063, 2064, 2065, 2068, 2069, 2070, 2074, 2075, 2076, 2078, 2079, 2081, 2109, 2110, 2111, 2112, 2113, 2116, 2119, 2120, 2121, 2123, 2124, 2125, 2126, 2131, 2183, 2184, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2218, 2219, 2220, 2221, 2223, 2224, 2225, 2379, 2380, 2382, 2410, 2411, 2412, 2413, 2415, 2501, 2547, 2593, 2662, 2663, 2709, 2710, 2712, 2713, 2714, 2715, 2716, 2723, 2724, 2725, 2726, 2727, 2729, 2730, 2731, 2732, 2733, 2734, 2735, 2736, 2737, 2741, 2743, 2744, 2745, 2746, 2747, 2748, 2749, 2750, 2751, 2752, 2753]
    features_df_new = data.iloc[:,feature_cols]
    return features_df_new
 

def train_model(model, submission=False):
    # Creates path to the training data and labels
    train_dataset = os.path.join("..", "data", "Train_call.txt")
    train_labels = os.path.join("..", "data", "Train_clinical.txt")

    # Determine file name for the saved model 
    # NOTE: this is universal to distinguish between the models. It will not save the model wit the requiered title model.pkl
    model_name = str(model) + "_model.pkl"
    if submission == True:
        model_name = "model.pkl"
    print(model_name)
    # load data, select best features and split the dataset
    X, y = load_data(train_dataset, train_labels)
    X = get_best_features(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)

    #  Initialise model with the hyperparameters obtained from hyperparamter tuning
    if model == "knn":     
        model = KNeighborsClassifier(leaf_size=1, n_neighbors=1, p=1) 
    elif model == "log_res":
        model = LogisticRegression(C=1.0, penalty='l2', solver='saga')
    elif model == "svc":
        model = SVC(C=1.5000000000000004, gamma="auto", kernel="rbf")
    else:
        print("Error: the model you want to train is not implemented.")

    # Fit the model and determine accuracy of the mode
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print("Accuracy score of the model: " + str(score))

    # Save trained model in a pickle file
    with open(model_name, 'wb') as file:
        pickle.dump(model, file)

if __name__ == '__main__':
    print("SVC accuracy:")
    train_model("svc")
    print("KNN accuracy:")
    train_model("knn")
    print("Logistic Regression accuracy:")
    train_model("log_res", True)