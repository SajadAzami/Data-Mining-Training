"""Linear Regression, 1/30/17, Sajad Azami"""

import data_preparation
import linear_regression
import numpy as np
import seaborn as sns
from matplotlib import gridspec
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import pandas as pd
import math
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression

__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'
sns.set_style("white")

train_full_X, train_full_Y = data_preparation.read_data('./data_set/train.csv', 6)
print('Data set Loaded!')

# In case any non-zero NA
train_full_X = train_full_X.fillna(0)

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

# Filling missing values with Gaussian Noise, 0.5 * N(mean_of_row, 1) + 0.5 * N(mean_of_class, 1)
for i in range(0, train_X.shape[0]):
    row = train_X.values[i]
    for j in range(0, len(row)):
        if row[j] == 0:
            # Considering only non-zero values in mean
            std_mean = sum(row) / len(row[np.nonzero(row)])
            if math.isnan(std_mean):
                std_mean = np.random.normal(10, scale=1)
            class_mean = sum(train_X[j]) / len(train_X[j].values[np.nonzero(train_X[j].values)])
            row[j] = round(0.5 * (abs(np.random.normal(std_mean, scale=1))) +
                           0.5 * (abs(np.random.normal(class_mean, scale=1))), 1)

# Filling missing values for test data using same way
for i in range(0, test_X.shape[0]):
    row = test_X.values[i]
    for j in range(0, len(row)):
        if row[j] == 0:
            # Considering only non-zero values in mean
            std_mean = sum(row) / len(row[np.nonzero(row)])
            if math.isnan(std_mean):
                std_mean = np.random.normal(10, scale=1)
            class_mean = sum(test_X[j]) / len(test_X[j].values[np.nonzero(test_X[j].values)])
            row[j] = round(0.5 * (abs(np.random.normal(std_mean, scale=1))) +
                           0.5 * (abs(np.random.normal(class_mean, scale=1))), 1)

# Scatter plot each feature vs label after filling missing values
fig = plt.figure()
gs = gridspec.GridSpec(2, 3)
counter = 0
for i in range(0, 2):
    for j in range(0, 3):
        counter += 1
        ax_temp = fig.add_subplot(gs[i, j])
        ax_temp.scatter(train_X.get(counter - 1), train_Y)
        ax_temp.title.set_text(('Feature ' + str(counter)))
plt.show()

# 1. Learning Linear Regression from Data
print('\nLinear Regression Model:')
linear_regression_model = LinearRegression()
linear_regression_line = linear_regression_model.fit(train_X, train_Y)
lr_predictions = linear_regression_line.predict(test_X)
title = 'Linear Regression'
RSS = sum((test_Y - lr_predictions) ** 2)
print('Linear Regression\nTest RSS: ' + str(RSS))
cv_risk = math.sqrt(sum(abs(cross_val_score(linear_regression_model,
                                            train_X, train_Y, scoring='mean_squared_error', cv=10))) / 10)
print('10-fold CV with RMSE: ' + str(cv_risk))
plt.plot(np.linspace(0, 40, 40), lr_predictions, 'r', label="predictions")
plt.plot(np.linspace(0, 40, 40), test_Y, label="real values")
plt.legend(loc='lower right')
plt.title(title)
plt.show()

# 2. Using Lasso Regression on Data
# Optimization Objective: (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
print('\nLasso Model:')
alpha = 0.1
lasso_model = Lasso(alpha=alpha)
lasso_line = lasso_model.fit(train_X, train_Y)
title = 'Alpha = ' + str(alpha)
lass_predictions = lasso_line.predict(test_X)
RSS = sum((test_Y - lass_predictions) ** 2)
print('Lasso with Lambda 0.0001\nTest RSS: ' + str(RSS))
cv_risk = math.sqrt(sum(abs(cross_val_score(lasso_model, train_X, train_Y, scoring='mean_squared_error', cv=10))) / 10)
print('10-fold CV with RMSE: ' + str(cv_risk))
plt.plot(np.linspace(0, 40, 40), lass_predictions, 'r', label="predictions")
plt.plot(np.linspace(0, 40, 40), test_Y, label="real values")
plt.legend(loc='lower right')
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
            title = 'Alpha = ' + str(alphas[counter])
        ax_temp = fig.add_subplot(gs[i, j])
        ax_temp.plot(np.linspace(0, 40, 40), predictions, 'r', label="predictions")
        ax_temp.plot(np.linspace(0, 40, 40), test_Y, label="real values")
        ax_temp.legend(loc='lower right')
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
