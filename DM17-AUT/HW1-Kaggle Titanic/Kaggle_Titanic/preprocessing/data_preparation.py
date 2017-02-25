"""Kaggle_Titanic, 11/7/16, Sajad Azami"""

import pandas as pd
import numpy as np
import seaborn as sns

sns.set_style('whitegrid')

__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'


# Reads train data from csv, returns pandas DF
def read_data(path):
    data = pd.read_csv(path, index_col=False)
    print('Read data successfully')
    return data


# Fills missing values with default values
# Each arg is a tuple ('feature_name', value)
def missing_handler(dataframe, *clm_value):
    for arg in clm_value:
        dataframe[arg[0]].fillna(arg[1])


# Splits labels from train data
# Each label arg is a tuple like ('label_name', to_be_dropped_flag)
def split_labels(data, *label_names):
    split = data[[label_names[0][0]]]
    for arg in label_names:
        if arg == label_names[0]:
            continue
        labels = data[[arg[0]]]
        split = np.concatenate((split, labels), axis=1)
    for arg in label_names:
        if arg[1]:
            data = data.drop(arg[0], 1)
    return data.values, split
