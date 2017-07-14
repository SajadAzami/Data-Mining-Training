import pandas as pd
from sklearn.metrics import v_measure_score, silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('HTRU_2.csv', header=None)
# data = data.sample(frac=0.2)
data_Y = data[8]
data_X = data.drop(8, axis=1)
v_scores = []
s_scores = []
for k in range(2, 21):
    print(k)
    pred = KMeans(n_clusters=k, random_state=28).fit_predict(data_X)
    v_scores.append(v_measure_score(data_Y, pred))
    s_scores.append(silhouette_score(data_X, pred))

plt.plot(range(2, 21), v_scores)
plt.show()
plt.plot(range(2, 21), s_scores)
plt.show()
