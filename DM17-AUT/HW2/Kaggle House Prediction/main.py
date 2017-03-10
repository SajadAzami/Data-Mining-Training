"""Kaggle House Prediction, 3/10/17, Sajad Azami"""

import data_preparation
import numpy as np
import seaborn as sns
from matplotlib import gridspec
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import pandas as pd
import math
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'
sns.set_style("white")

train_full_X, train_full_Y = data_preparation.read_data('./data_set/train.csv', 'SalePrice')
print('Data set Loaded!, Size: ' + str(train_full_X.shape))

print('\nMissing Status:')
print(data_preparation.show_missing(train_full_X))
# Dropping features with huge number of NAs: [PoolQC, Fence, MiscFeature, Alley]
train_full_X = train_full_X.drop(['PoolQC', 'Fence', 'MiscFeature', 'Alley'], axis=1)

# Split train and test data
train_X = train_full_X[:1060]
train_Y = train_full_Y[:1060]
test_X = train_full_X[1060:]
test_Y = train_full_Y[1060:]

print('Train data size:', train_X.shape)
print('Test data size:', test_X.shape)
