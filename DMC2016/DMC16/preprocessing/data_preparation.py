"""DMC16, 11/9/16, Sajad Azami"""

import pandas as pd
import numpy as np

__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'


# Returns data in DataFrames
def read_data(PATH):
    data = pd.read_csv(PATH, sep=';', nrows=200000)
    print('Read data successfully')
    print('Rows: ', len(data))
    print(data.info())
    return data


train_data_set = read_data('../data_set/orders_train.txt')


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


train_data, train_labels = split_labels(train_data_set,
                                        ('orderID', False), ('articleID', False),
                                        ('colorCode', False), ('sizeCode', False),
                                        ('returnQuantity', True))
print(train_labels)
