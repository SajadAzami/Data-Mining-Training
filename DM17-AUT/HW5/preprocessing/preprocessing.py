import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style('dark')


def pre_process(data_train_X, data_train_Y):
    # Filling missing values for feature amount with mean
    data_train_X['amount'] = data_train_X['amount'].replace(0, np.NaN)
    data_train_X['amount'].fillna(data_train_X['amount'].mean(), inplace=True)

    # Dummy zip feature(Dropped for now)
    # TODO cluster and dummy
    print(len(data_train_X['zip'].unique()))
    data_train_X = data_train_X.drop('zip', axis=1)  # Dropping for now

    # Dummy state feature
    data_train_X = pd.concat([data_train_X, pd.get_dummies(data_train_X['state'])], axis=1)
    data_train_X = data_train_X.drop('state', axis=1)

    # Feature hour_a and hour_b
    data_train_X['hour_a'].corr(data_train_X['hour_b'])
    # plt.scatter(data_train_X['hour_a'], data_train_X['hour_b'])
    # plt.show()
    data_train_X = data_train_X.drop('hour_b', axis=1)

    # Feature customerAttr_a
    # data_train_X['customerAttr_a'] = data_train_X['customerAttr_a'] - 1234567890123456
    data_train_X = data_train_X.drop('customerAttr_a', axis=1)

    # Feature total
    print(data_train_X['amount'].corr(data_train_X['total']))
    data_train_X = data_train_X.drop('total', axis=1)

    # Feature customerAttr_b
    # TODO cluster and dummy
    def substring_after(s, delim):
        return s.partition(delim)[2].partition('.')[2]

    data_train_X['customerAttr_b'] = data_train_X['customerAttr_b'].apply(substring_after, delim='@')
    data_train_X = data_train_X.drop('customerAttr_b', axis=1)  # Dropping for now

    return data_train_X, data_train_Y
