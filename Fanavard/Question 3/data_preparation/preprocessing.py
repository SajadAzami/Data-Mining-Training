"""Question 3, 11/23/16, Sajad Azami"""

import pandas as pd
import numpy as np
import re

__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'


# Reads train data from csv, returns pandas DF
def read_train():
    data = pd.read_csv('../data_set/data_train.csv')
    return data


# Reads train
def read_train_split_time():
    data = pd.read_csv('../data_set/data_train.csv')
    time = data.get('TIME')
    data.drop('TIME', axis=1)
    hours = []
    minutes = []
    seconds = []
    for i in range(0, len(time)):
        time_str = re.split(':', time[i])
        hours.append(time_str[0])
        minutes.append(time_str[1])
        seconds.append(time_str[2])
    data['HOUR'] = hours
    data['MINUTE'] = minutes
    data['SECOND'] = seconds
    return data


def read_test():
    return pd.read_csv('../data_set/data_test.csv')


def get_k_fold_train_test():
    data = read_train()
    test_index = np.random.choice(data.index, int(len(data.index) / 10), replace=False)

    test = data.loc[test_index]
    train = data.loc[~data.index.isin(test_index)]

    return train, test


def get_frauds(data):
    frauds = []
    for i in data.get_values():
        if i[9] == 1:
            frauds.append(i)
            print(i)

    return frauds
