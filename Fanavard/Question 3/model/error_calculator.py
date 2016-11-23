"""Question 3, 11/24/16, Sajad Azami"""

import pandas as pd

__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'


def get_accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)


def get_sensitivity(TP, TN, FP, FN):
    return TP / (TP + FN)


def get_specificity(TP, TN, FP, FN):
    return TN / (TN + FP)


def get_precision(TP, TN, FP, FN):
    return TP / (TP + FP)


def get_F1score(TP, TN, FP, FN):
    return (2 * TP) / (2 * TP + FP + FN)


submission = pd.read_csv('./submission.csv')
validation = pd.read_csv('./validation.csv')

# Positive Class: Fraud, Negative Class: Not Fraud
TP = 0  # Prediction:1, Value:1
TN = 0  # Prediction:0, Value:0
FP = 0  # Prediction:1, Value:0
FN = 0  # Prediction:0, Value:1
for index_sub, item_sub in submission.iterrows():
    print(item_sub['Is Fraud'])
    for index_val, item_val in validation.iterrows():
        print(item_val['Is Fraud'])
        if item_sub['Transaction ID'] == item_val['Transaction ID']:
            if item_sub['Is Fraud'] == 0 and item_val['Is Fraud'] == 1:
                FN += 1
                break
            if item_sub['Is Fraud'] == 1 and item_val['Is Fraud'] == 0:
                FP += 1
                break
            if item_sub['Is Fraud'] == 0 and item_val['Is Fraud'] == 0:
                TN += 1
                break
            if item_sub['Is Fraud'] == 1 and item_val['Is Fraud'] == 1:
                TP += 1
                break
