"""Kaggle_Titanic, 11/7/16, Sajad Azami"""

import pandas as pd
import numpy as np

__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'


def read_data(PATH):
    data = pd.read_csv(PATH)
    print('Read data successfully')
    print('Rows: ', len(data))
    return data


train_data_set = read_data('./data_set/train.csv')
print(train_data_set.head())
