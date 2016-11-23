"""Question 3, 11/23/16, Sajad Azami"""

import pandas as pd

__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'


def read_train():
    data = pd.read_csv('../data_set/data_train.csv')
    # TODO split time
    return data


def read_test():
    return pd.read_csv('../data_set/data_test.csv')


def get_frauds(data):
    frauds = []
    for i in data.get_values():
        if i[9] == 1:
            frauds.append(i)
            print(i)

    return frauds


print(get_frauds(read_train()))
