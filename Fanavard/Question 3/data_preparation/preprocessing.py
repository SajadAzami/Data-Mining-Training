"""Question 3, 11/23/16, Sajad Azami"""

import pandas as pd
import re

__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'


def read_train():
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


def get_frauds(data):
    frauds = []
    for i in data.get_values():
        if i[9] == 1:
            frauds.append(i)
            print(i)

    return frauds


print(read_train().head)
