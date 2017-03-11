"""Kaggle House Prediction, 3/10/17, Sajad Azami"""

import data_preparation
import numpy as np
import seaborn as sns
from matplotlib import gridspec
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import pandas as pd
import math
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'
sns.set_style("white")


# Looking at categorical values
def cat_exploration(column, data):
    return data[column].value_counts()


# Imputing the missing values
def cat_imputation(column, value, data):
    data.loc[data[column].isnull(), column] = value


train_full_X, train_full_Y = data_preparation.read_data('./data_set/train.csv', 'SalePrice')
test_full_X = pd.read_csv('./data_set/test.csv')
print('Data set Loaded!, Size: ' + str(train_full_X.shape))

# print('\nMissing Status:')
# print(data_preparation.show_missing(train_full_X))

# IMPUTATION
# Dropping features with huge number of NAs: [PoolQC, Fence, MiscFeature]
# train_full_X = train_full_X.drop(['PoolQC', 'Fence', 'MiscFeature'], axis=1)
# test_full_X = test_full_X.drop(['PoolQC', 'Fence', 'MiscFeature'], axis=1)
cat_imputation('Alley', 'None', train_full_X)
cat_imputation('Alley', 'None', test_full_X)
cat_imputation('MasVnrType', 'None', train_full_X)
cat_imputation('MasVnrType', 'None', test_full_X)
cat_imputation('MasVnrArea', 0.0, train_full_X)
cat_imputation('MasVnrArea', 0.0, test_full_X)
basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2']
for cols in basement_cols:
    if 'FinSF' not in cols:
        cat_imputation(cols, 'None', train_full_X)
        cat_imputation(cols, 'None', test_full_X)

# Impute most frequent value
cat_imputation('Electrical', 'SBrkr', train_full_X)
cat_imputation('Electrical', 'SBrkr', test_full_X)
cat_imputation('FireplaceQu', 'None', train_full_X)
cat_imputation('FireplaceQu', 'None', test_full_X)

garage_cols = ['GarageType', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea']
# Garage Imputation
for cols in garage_cols:
    if train_full_X[cols].dtype == np.object:
        cat_imputation(cols, 'None', train_full_X)
    else:
        cat_imputation(cols, 0, train_full_X)
    if test_full_X[cols].dtype == np.object:
        cat_imputation(cols, 'None', test_full_X)
    else:
        cat_imputation(cols, 0, test_full_X)

cat_imputation('PoolQC', 'None', train_full_X)
cat_imputation('PoolQC', 'None', test_full_X)
cat_imputation('Fence', 'None', train_full_X)
cat_imputation('Fence', 'None', test_full_X)
cat_imputation('MiscFeature', 'None', train_full_X)
cat_imputation('MiscFeature', 'None', test_full_X)

train_full_X = train_full_X.fillna(train_full_X.mean())
test_full_X = test_full_X.fillna(test_full_X.mean())

# Converting categorical to numeric
cols_to_transform = train_full_X.columns.difference(train_full_X._get_numeric_data().columns)
train_full_X = pd.get_dummies(train_full_X, columns=cols_to_transform)
test_full_X = pd.get_dummies(test_full_X, columns=cols_to_transform)

# Split train and test data
train_X = train_full_X[:1060]
train_Y = train_full_Y[:1060]
test_X = train_full_X[1060:]
test_Y = train_full_Y[1060:]

print('Train data size:', train_X.shape)
print('Test data size:', test_X.shape)

# # 1. Random Forest model
# forest = RandomForestClassifier(max_features='sqrt')
# parameter_grid = {
#     'max_depth': [4, 5, 6, 7, 8],
#     'n_estimators': [200, 210, 240, 250],
#     'criterion': ['gini', 'entropy']
# }
#
# cross_validation = StratifiedKFold(train_Y, n_folds=5)
# grid_search = GridSearchCV(forest,
#                            param_grid=parameter_grid,
#                            cv=cross_validation)
# grid_search.fit(train_X, train_Y)
# print('Best score: {}'.format(grid_search.best_score_))
# print('Best parameters: {}'.format(grid_search.best_params_))
# Y_pred = grid_search.predict(test_full_X).astype(int)
# submission = pd.DataFrame(Y_pred)
# submission.to_csv(path_or_buf='random_forest.csv', index=False)

# 2. Using Lasso Regression on Data
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
