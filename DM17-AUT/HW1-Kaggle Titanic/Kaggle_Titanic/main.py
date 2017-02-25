"""Kaggle_Titanic, 2/22/17, Sajad Azami"""

from preprocessing import data_preparation as dp
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import gridspec
import matplotlib.pyplot as plt

sns.set_style('white')
__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'


# Scatter plot each feature vs label, n*m subplot
def scatter_plot_feature_vs_label(data, label, n, m):
    fig = plt.figure()
    gs = gridspec.GridSpec(n, m)
    counter = 0
    for i in range(0, n):
        for j in range(0, m):
            ax_temp = fig.add_subplot(gs[i, j])
            ax_temp.scatter(data[data.columns.values[counter]].values, label)
            ax_temp.title.set_text(('Feature ' + str(data.columns.values[counter])))
            counter += 1
            if counter >= len(data.columns.values):
                break
    plt.show()


train_data_set = dp.read_data('./data_set/train.csv')
train_label = train_data_set['Survived']
train_data = train_data_set.drop('Survived', axis=1)
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
scatter_plot_feature_vs_label(train_data[['Age', 'PassengerId', 'Pclass', 'Fare', 'SibSp', 'Parch']], train_label, 3, 2)
print(train_data.info())
