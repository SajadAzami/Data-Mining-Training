"""Question 3, 11/26/16, Sajad Azami"""

import pandas as pd
import xgboost as xgb

__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'

# Load the data, using 1/10 of the data as validation
train_df = pd.read_csv('../data_set/oversampled_data.csv')
test_df = pd.read_csv('../data_set/data_test.csv')
print(train_df.info())
print(test_df.info())

feature_columns_to_use = ['PS ID', 'PROVINCE ID', 'COUNTY ID', 'Date', 'AMOUNT']

# Join the features from train and test together before imputing missing values,
# in case their distribution is slightly different
big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])

# Prepare the inputs for the model
train_X = big_X[0:train_df.shape[0]].as_matrix()
test_X = big_X[train_df.shape[0]::].as_matrix()
train_y = train_df['Is Fraud']

# Learn model
print('Model Learning Started')
print('Train Size: ', len(train_df))
print('Test Size: ', len(test_df))

# Learning with XGBClassifier
gbm = xgb.XGBClassifier(max_depth=5, n_estimators=300,
                        learning_rate=0.05).fit(train_X, train_y)
preds = gbm.predict(test_X)

# Create submission file
submission = pd.DataFrame({'Transaction ID': test_df['Transaction ID'],
                           'Is Fraud': preds})

submission.to_csv('submission_final.csv', index=False)
