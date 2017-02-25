"""Kaggle_Titanic, 2/26/17, Sajad Azami"""

from matplotlib import gridspec
import matplotlib.pyplot as plt
import pandas as pd

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


# Bar plot each feature vs label, n*m subplot
def bar_plot_feature_vs_label(data, label, target, n, m):
    fig = plt.figure()
    gs = gridspec.GridSpec(n, m)
    counter = 0
    for i in range(0, n):
        for j in range(0, m):
            ax_temp = fig.add_subplot(gs[i, j])
            one_label = data[data[label] == 1][target[counter]].value_counts()
            zero_label = data[data[label] == 0][target[counter]].value_counts()
            df = pd.DataFrame([one_label, zero_label])
            df.index = [label, 'not' + label]
            df.plot(kind='bar', stacked=True, figsize=(15, 8), title='Feature ' + str(target[counter]))
            counter += 1
            if counter >= len(target):
                break
    plt.show()
