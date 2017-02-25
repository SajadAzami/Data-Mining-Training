"""Kaggle_Titanic, 2/22/17, Sajad Azami"""

from preprocessing import data_preparation as dp
import pandas as pd
import numpy as np
import seaborn as sns
from visualization import scatter
import matplotlib.pyplot as plt

sns.set_style('white')
__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'

# Loading dataset
train_data_set = dp.read_data('./data_set/train.csv')
test_data_set = dp.read_data('./data_set/test.csv')

# Filling missing values
train_data_set['Age'] = train_data_set['Age'].fillna(train_data_set['Age'].median())
test_data_set['Age'] = test_data_set['Age'].fillna(test_data_set['Age'].median())
train_data_set["Fare"] = train_data_set["Fare"].fillna(train_data_set["Fare"].median())
test_data_set["Fare"] = test_data_set["Fare"].fillna(test_data_set["Fare"].median())
# convert from float to int
test_data_set['Fare'] = test_data_set['Fare'].astype(int)
test_data_set['Fare'] = test_data_set['Fare'].astype(int)


# Scatter plot numerical data
scatter.bar_plot_feature_vs_label(train_data_set, 'Survived',
                                  ['Pclass', 'SibSp', 'Parch'], 2, 2)

# drop unnecessary columns, these columns won't be useful in analysis and prediction
train_data_set = train_data_set.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
test_data_set = test_data_set.drop(['Name', 'Ticket'], axis=1)
