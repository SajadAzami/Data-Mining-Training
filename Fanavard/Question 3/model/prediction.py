"""DMC16, 11/9/16, Sajad Azami"""

import xgboost as xgb
import pandas as pd
from model import error_calculator
from data_preparation import preprocessing

__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'


# Creates a submission file from submission DF
def submission_to_csv(submission_df, name):
    submission_df.to_csv(name, index=False)


# Rounds predictions to 0 or 1 (>0.8)
def round_predictions(preds):
    for i in range(0, len(preds)):
        if preds[i] >= 0.8:
            preds[i] = 1
        else:
            preds[i] = 0
    return preds


def train_XGBTrain():
    train_df, test_df = preprocessing.get_k_fold_train_test('../data_set/oversampled_data.csv')

    xg_train = xgb.DMatrix(train_df.values, train_df['Is Fraud'].values)
    xg_test = xgb.DMatrix(test_df.values, test_df['Is Fraud'].values)

    validation_df = test_df.loc[:, ['Is Fraud', 'Transaction ID']]

    # Learn Model
    print('Model Learning Started')
    print('Train Size: ', len(train_df))
    print('Test Size: ', len(test_df))
    param = {'silent': 1, 'objective': 'binary:logistic', 'eval_metric': 'error'}

    num_round = 10
    eval_list = [(xg_train, 'train')]

    bst = xgb.train(param, xg_train, num_round, eval_list)
    preds = bst.predict(xg_test)

    preds = round_predictions(preds)
    preds = preds.astype(int)

    # Create submission file
    submission = pd.DataFrame({'Transaction ID': test_df['Transaction ID'],
                               'Is Fraud': preds})

    submission_to_csv(submission, 'submission_XGBTrain.csv')

    errors = error_calculator.get_errors(submission, validation_df)
    print('Accuracy:', error_calculator.get_accuracy(errors))
    return preds


def train_XGBClassifier():
    # Load the data, using 1/10 of the data as validation
    train_df, test_df = preprocessing.get_k_fold_train_test('../data_set/oversampled_data.csv')

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

    validation_df = test_df.loc[:, ['Is Fraud', 'Transaction ID']]
    validation_df.to_csv('validation.csv', index=False)

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

    submission_to_csv(submission, 'submission_XGBClassifier.csv')

    errors = error_calculator.get_errors(submission, validation_df)
    print('Accuracy:', error_calculator.get_accuracy(errors))
    return preds


train_df, test_df = preprocessing.get_k_fold_train_test('../data_set/oversampled_data.csv')

predictions_XGBClassifier = train_XGBClassifier()
predictions_XGBTrain = train_XGBTrain()
