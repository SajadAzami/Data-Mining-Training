"""DMC16, 11/17/16, Sajad Azami"""

import pandas as pd

__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'

submission = pd.read_csv('./submission.csv')
real_class = pd.read_csv('../data_set/realclass_DMC_2016.txt', sep=";")

error = 0
print(len(submission))
print(len(real_class))
for i in range(len(submission)):
    error += abs(real_class['returnQuantity'].get(i) - submission['returnQuantity'].get(i))
print(error)
