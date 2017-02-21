"""Kaggle_Titanic, 11/7/16, Sajad Azami"""

import pandas as pd
import numpy as np
import seaborn as sns

sns.set_style('whitegrid')

__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'


# Returns data in DataFrames
def read_data(PATH):
    data = pd.read_csv(PATH)
    print('Read data successfully')
    print('Rows: ', len(data))
    print(data.info())
    return data


# Fills missing values with default values
# Each arg is a tuple ('feature_name', value)
def missing_handler(dataframe, *clm_value):
    for arg in clm_value:
        dataframe[arg[0]].fillna(arg[1])


# Splits labels from train data
# Each label arg is a tuple like ('label_name', to_be_droped_flag)
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


train_data_set = read_data('./data_set/train.csv')
missing_handler(train_data_set, ('Embarked', 'S'))
sns.factorplot('Embarked', 'Survived', data=train_data_set, size=4, aspect=3)
train_data, train_labels = split_labels(train_data_set, ('PassengerId', False), ('Survived', True))
