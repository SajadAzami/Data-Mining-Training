"""Kaggle_Titanic, 11/7/16, Sajad Azami"""

import pandas as pd
import numpy as np

__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'


# Returns data in DataFrames
def read_data(PATH):
    data = pd.read_csv(PATH)
    print('Read data successfully')
    print('Rows: ', len(data))
    # print(data.describe())
    return data


train_data_set = read_data('./data_set/train.csv')


# Splits labels from train data
# Each label arg is a tuple like ('label_name', to_be_droped_flag)
def split_labels(data, *label_names):
    splitted = data[[label_names[0][0]]]
    for arg in label_names:
        if arg == label_names[0]:
            continue
        labels = data[[arg[0]]]
        splitted = np.concatenate((splitted, labels), axis=1)
    for arg in label_names:
        if arg[1]:
            data = data.drop(arg[0], 1)
    return data.values, splitted


print(split_labels(train_data_set, ('PassengerId', False), ('Survived', True)))
