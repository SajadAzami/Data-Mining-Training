"""Kaggle_Titanic, 11/8/16, Sajad Azami"""

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'


# TODO To be completed
# Implements PCA on String Typed Data
def pca_on_string(data):
    print(data.astype(str).values)
    pca = PCA(n_components=11)
