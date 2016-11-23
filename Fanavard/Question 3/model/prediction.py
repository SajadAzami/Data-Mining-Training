"""DMC16, 11/9/16, Sajad Azami"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import TransformerMixin
from data_preparation import preprocessing

__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'

# Load the data
train_df = preprocessing.read_train()
test_df = preprocessing.read_test()

print(train_df.info())
print(test_df.info())


# Use this if you want to train and test on train data
# train_df = train_df[680001:860000]
# test_df = train_df[2000001:2200000]


class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
                              index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


feature_columns_to_use = ['PS ID', 'PROVINCE ID', 'COUNTY ID', 'Date', 'AMOUNT']

# Join the features from train and test together before imputing missing values,
# in case their distribution is slightly different
big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])
big_X_imputed = DataFrameImputer().fit_transform(big_X)

# Prepare the inputs for the model
train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()
test_X = big_X_imputed[train_df.shape[0]::].as_matrix()
train_y = train_df['Is Fraud']

# Learn model
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)
predictions = gbm.predict(test_X)

# Create submission file
submission = pd.DataFrame({'PS ID': test_df['PS ID'],
                           'PROVINCE ID': test_df['PROVINCE ID'],
                           'COUNTY ID': test_df['COUNTY ID'],
                           'Date': test_df['Date'],
                           'AMOUNT': test_df['AMOUNT'],
                           'Is Fraud': predictions})
submission.to_csv("submission.csv", index=False)
