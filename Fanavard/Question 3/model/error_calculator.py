"""Question 3, 11/24/16, Sajad Azami"""

__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'


def get_accuracy(errors):
    print('TP: ', errors[0], 'TN: ', errors[1],
          'FP: ', errors[2], 'FN: ', errors[3])
    return (errors[0] + errors[1]) / (errors[0] + errors[1] + errors[2] + errors[3])


def get_sensitivity(errors):
    return errors[0] / (errors[0] + errors[3])


def get_specificity(errors):
    return errors[1] / (errors[1] + errors[2])


def get_precision(errors):
    return errors[0] / (errors[0] + errors[2])


def get_F1score(errors):
    return (2 * errors[0]) / (2 * errors[0] + errors[2] + errors[3])


# Calculates errors from submission and validation DF
def get_errors(submission_df, validation_df):
    # Positive Class: Fraud, Negative Class: Not Fraud
    TP = 0  # Prediction:1, Value:1
    TN = 0  # Prediction:0, Value:0
    FP = 0  # Prediction:1, Value:0
    FN = 0  # Prediction:0, Value:1

    diff = submission_df['Is Fraud'].values - validation_df['Is Fraud'].values
    multiply = submission_df['Is Fraud'].values * validation_df['Is Fraud'].values

    for i in range(0, len(diff)):
        if diff[i] == -1:
            FN += 1
        if diff[i] == 1:
            FP += 1
    T = len(diff) - FP - FN

    for i in range(0, len(multiply)):
        if multiply[i] == 1:
            TP += 1
    TN = T - TP
    errors = (TP, TN, FP, FN)
    return errors
