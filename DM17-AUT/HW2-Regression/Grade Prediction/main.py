"""Linear Regression, 1/30/17, Sajad Azami"""

import data_preparation
import linear_regression
import numpy as np
import seaborn as sns
from matplotlib import gridspec
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import pandas as pd

__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'
sns.set_style("white")

train_full_X, train_full_Y = data_preparation.read_data('./data_set/train.csv', 6)
print('Data set Loaded!')

# Split train and test data
train_X = train_full_X[:200]
train_Y = train_full_Y[:200]
test_X = train_full_X[200:]
test_Y = train_full_Y[200:]

print('Train data size:', len(train_X))
print('Test data size:', len(test_X))

# Scatter plot each feature vs label
fig = plt.figure()
gs = gridspec.GridSpec(2, 3)
counter = 0
for i in range(0, 2):
    for j in range(0, 3):
        counter += 1
        ax_temp = fig.add_subplot(gs[i, j])
        ax_temp.scatter(train_full_X.get(counter - 1), train_full_Y)
        ax_temp.title.set_text(('Feature ' + str(counter)))
plt.show()

# Filling missing values with Gaussian Noise, N(mean_of_row, 1)
for i in range(0, train_X.shape[0]):
    row = train_X.values[i]
    for j in range(0, len(row)):
        if row[j] == 0:
            row[j] = abs(np.random.normal(sum(row) / len(row), scale=1))
# Filling missing values of test data with the same way
for i in range(0, test_X.shape[0]):
    test_row = test_X.values[i]
    if test_row[j] == 0:
        test_row[j] = abs(np.random.normal(sum(test_row) / len(test_row), scale=1))

# Scatter plot each feature vs label after filling missing values
fig = plt.figure()
gs = gridspec.GridSpec(2, 3)
counter = 0
for i in range(0, 2):
    for j in range(0, 3):
        counter += 1
        ax_temp = fig.add_subplot(gs[i, j])
        ax_temp.scatter(train_full_X.get(counter - 1), train_full_Y)
        ax_temp.title.set_text(('Feature ' + str(counter)))
plt.show()

# Using Lasso Regression on Data
# Optimization Objective: (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
alpha = 0.000001
lasso_model = Lasso(alpha=alpha)
lasso_line = lasso_model.fit(train_X, train_Y)
title = 'Alpha = ' + str(alpha) + '\nRed: Lasso Prediction, Blue: Real Values'
plt.plot(np.linspace(0, 40, 40), lasso_line.predict(test_X), 'r', label="predictions")
plt.plot(np.linspace(0, 40, 40), test_Y, label="real values")
plt.legend(loc='upper left')
plt.title(title)
plt.show()

# Testing for different alphas
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
counter = 0
fig = plt.figure()
gs = gridspec.GridSpec(2, 3)
for i in range(0, 2):
    for j in range(0, 3):
        lasso_model = Lasso(alpha=alphas[counter])
        lasso_line = lasso_model.fit(train_X, train_Y)
        predictions = lasso_line.predict(test_X)
        if counter == 0:
            RSS = sum((test_Y - predictions) ** 2)
            print('RSS: ' + str(RSS))
        title = 'Alpha = ' + str(alphas[counter])
        ax_temp = fig.add_subplot(gs[i, j])
        ax_temp.plot(np.linspace(0, 40, 40), predictions, 'r', label="predictions")
        ax_temp.plot(np.linspace(0, 40, 40), test_Y, label="real values")
        ax_temp.legend(loc='upper left')
        ax_temp.title.set_text(title)
        counter += 1
plt.show()
#
# # Predicting the final submission file with the best model
# # Filling missing values with Gaussian Noise, N(mean_of_row, 1)
# final_data = pd.read_csv('../data_set/Dataset2_Unlabeled.csv', header=None)
# for i in range(0, final_data.shape[0]):
#     row = final_data.values[i]
#     for j in range(0, len(row)):
#         if row[j] == 0:
#             row[j] = abs(np.random.normal(sum(row) / len(row), scale=1))
# lasso_model = Lasso(alpha=0.0001)
# lasso_line = lasso_model.fit(train_data_1, train_label_1)
# predictions = lasso_line.predict(final_data)
# predictions.tofile('predictions.csv', sep=',\n')
