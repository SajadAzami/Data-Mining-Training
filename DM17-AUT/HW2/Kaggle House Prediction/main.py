"""Kaggle House Prediction, 3/10/17, Sajad Azami"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression

import data_preparation

__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'
sns.set_style("white")


# Looking at categorical values
def cat_exploration(column, data):
    return data[column].value_counts()


# Imputing the missing values
def cat_imputation(column, value, data):
    data.loc[data[column].isnull(), column] = value


def encode_field(dataframe_train, dataframe_test, col_name):
    encoder = preprocessing.LabelEncoder()

    for i in col_name:
        dataframe_train = dataframe_train.fillna(dataframe_train[i].value_counts().index[0])
        dataframe_test = dataframe_test.fillna(dataframe_test[i].value_counts().index[0])

        encoder.fit(dataframe_train[i].values)
        dataframe_train[i] = encoder.transform(dataframe_train[i].values)
        dataframe_test[i] = encoder.transform(dataframe_test[i].values)
    return dataframe_train, dataframe_test


train_full_X, train_full_Y = data_preparation.read_data('./data_set/train.csv', 'SalePrice')
test_full_X = pd.read_csv('./data_set/test.csv')
submission_ids = test_full_X['Id']

print('Data set Loaded!\nTrain Shape: ' + str(train_full_X.shape))
print('Final Test Shape: ' + str(test_full_X.shape))


# print('\nMissing Status:')
# print(data_preparation.show_missing(train_full_X))

# IMPUTATION
# Dropping features with huge number of NAs: [PoolQC, Fence, MiscFeature]
# train_full_X = train_full_X.drop(['PoolQC', 'Fence', 'MiscFeature'], axis=1)
# test_full_X = test_full_X.drop(['PoolQC', 'Fence', 'MiscFeature'], axis=1)
def fill_na(dataframe):
    for column in dataframe.columns:
        if dataframe[column].count() <= len(dataframe[column]) * 0.7:
            del dataframe[column]
            continue
        if type(dataframe[column]) is np.int64:
            dataframe[column].fillna(dataframe[column].median(), inplace=True)
        elif type(dataframe[column]) is np.float64:
            dataframe[column].fillna(dataframe[column].median(), inplace=True)
        else:
            dataframe[column].fillna(dataframe[column].value_counts().idxmax(), inplace=True)


fill_na(train_full_X)
fill_na(test_full_X)

# Converting categorical to numeric
categorical_columns = train_full_X.columns.difference(train_full_X._get_numeric_data().columns)
train_full_X, test_full_X = encode_field(train_full_X, test_full_X, categorical_columns)

# Select useful features
# This method didn't work well, so it was commented
# lsvc = LassoCV(n_alphas=1).fit(train_X, train_Y)
# model = SelectFromModel(lsvc, prefit=True)
# train_X = model.transform(train_X)
# print('Shape after transform:' + str(train_X.shape))
# test_X = model.transform(test_X)
# test_full_X = model.transform(test_full_X)

train_full_X = train_full_X.drop('Id', axis=1)
test_full_X = test_full_X.drop('Id', axis=1)

# cols_to_transform = train_full_X.columns.difference(train_full_X._get_numeric_data().columns)
# train_full_X = pd.get_dummies(train_full_X, columns=cols_to_transform)
# test_full_X = pd.get_dummies(test_full_X, columns=cols_to_transform)

# Split train and test data
train_X = train_full_X[:1060]
train_Y = train_full_Y[:1060]
test_X = train_full_X[1060:]
test_Y = train_full_Y[1060:]

print('Train data size:', train_X.shape)
print('Test data size:', test_X.shape)
print(test_full_X.shape)
print(train_X.shape)

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
n = len(lr_predictions)
plt.plot(np.linspace(0, n, n), lr_predictions, 'r', label="predictions")
plt.plot(np.linspace(0, n, n), test_Y, label="real values")
plt.legend(loc='lower right')
plt.title(title)
plt.show()

# 2. Random Forest model
print('\nRandom Forrest Model:')
forest = RandomForestClassifier(max_features='sqrt')
parameter_grid = {
    'max_depth': [40],
    'n_estimators': [250],
    'criterion': ['entropy']
}

cross_validation = StratifiedKFold(train_Y, n_folds=10)
grid_search = GridSearchCV(forest,
                           param_grid=parameter_grid,
                           cv=cross_validation)
grid_search.fit(train_X, train_Y)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
Y_pred = grid_search.predict(test_X).astype(int)
title = 'Random Forrest Model'
plt.plot(np.linspace(0, 400, 400), Y_pred, 'r', label="predictions")
plt.plot(np.linspace(0, 400, 400), test_Y, label="real values")
RSS = sum((test_Y - Y_pred) ** 2)
print('Random Forrest\nTest RSS: ' + str(RSS))
plt.legend(loc='lower right')
plt.title(title)
plt.show()

# 3. Using Lasso Regression on Data
# Optimization Objective: (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

print('\nLasso Model:')
alpha = 0.1
lasso_model = Lasso(alpha=alpha)
lasso_line = lasso_model.fit(train_X, train_Y)
title = 'Lasso with Alpha = ' + str(alpha)
lasso_predictions = lasso_line.predict(test_X)
RSS = sum((test_Y - lasso_predictions) ** 2)
print('Lasso with Lambda 0.1\nTest RSS: ' + str(RSS))
cv_risk = math.sqrt(sum(abs(cross_val_score(lasso_model, train_X, train_Y, scoring='mean_squared_error', cv=10))) / 10)
print('10-fold CV with RMSE: ' + str(cv_risk))
n = len(lasso_predictions)
plt.plot(np.linspace(0, n, n), lasso_predictions, 'r', label="predictions")
plt.plot(np.linspace(0, n, n), test_Y, label="real values")
plt.legend(loc='lower right')
plt.title(title)
plt.show()

# Predicting the final submission file with the best model, Random Forrest
lasso_line = lasso_model.fit(train_full_X, train_full_Y)
test_prediction = lasso_line.predict(test_full_X).astype(int)
print(test_prediction)
submission = pd.DataFrame({'Id': submission_ids, 'SalePrice': test_prediction})
submission.to_csv('lasso_predictions.csv', index=None)
