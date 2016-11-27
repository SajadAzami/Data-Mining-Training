"""Question 3, 11/27/16, Sajad Azami"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'

sns.set_style('darkgrid')
data = pd.read_csv('../data_set/data_train.csv')

data = data.sort_values('Is Fraud', ascending=False)


def plot_vs_feature(feature):
    county_id = np.zeros(data.max()[feature] + 10)
    for index, row in data.iterrows():
        if row['Is Fraud'] == 1:
            id_temp = row[feature]
            county_id[id_temp] += 1
        else:
            break
    plt.plot(county_id)
    plt.title(feature)
    plt.show()


plot_vs_feature('COUNTY ID')
plot_vs_feature('PROVINCE ID')
plot_vs_feature('Date')
