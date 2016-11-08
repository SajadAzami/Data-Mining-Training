"""Kaggle_Titanic, 11/7/16, Sajad Azami"""

import pandas as pd
import numpy as np

__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'


# Returns data in DataFrames
def read_data(PATH):
    data = pd.read_csv(PATH)
    print('Read data successfully')
    print('Rows: ', len(data))
    return data


train_data_set = read_data('./data_set/train.csv')
labels = train_data_set[['PassengerId', 'Survived']]

train_data_set = train_data_set.drop('Survived', 1)
