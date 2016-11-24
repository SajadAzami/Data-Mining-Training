"""Question 3, 11/23/16, Sajad Azami"""

import pandas as pd
import numpy as np
import re

__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'


# Reads train data from csv, returns pandas DF
def read_train():
    data = pd.read_csv('../data_set/data_train.csv')
    return data


# Reads train and splits the TIME to HOUR, MINUTE and SECOND
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


#
def read_test():
    return pd.read_csv('../data_set/data_test.csv')


def get_k_fold_train_test():
    data = read_train()
    test_index = np.random.choice(data.index, int(len(data.index) / 10), replace=False)

    test = data.loc[test_index]
    train = data.loc[~data.index.isin(test_index)]

    return train, test


# Gets a DF of train data with all frauds copied n times in train data
def get_frauds(data):
    # TODO Duplicate Fraudual data using pandas own methods
    frame = []
    for index, row in data.iterrows():
        print(index)
        if row['Is Fraud'] == 1:
            frame.append(row.values)
    new_data_list = np.vstack((np.array(data.values).transpose(), np.array(frame).transpose()))
    new_data_df = pd.DataFrame(new_data_list)
    return new_data_df


print(len(read_train()))
get_frauds(read_train()).to_csv('oversampled_data.csv')
