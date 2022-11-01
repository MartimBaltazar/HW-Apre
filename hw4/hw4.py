from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, metrics, cluster, mixture
from sklearn.preprocessing import MinMaxScaler

def purity_score(y_true, y_pred):
    # compute contingency/confusion matrix
    confusion_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix) 

data = loadarff('pd_speech.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].str.decode('utf-8')
X = df.drop('class', axis=1)
y = df['class']

seeds = [1,2,3]

# parameterize clustering
kmeans_algo1 = cluster.KMeans(n_clusters=3, random_state=seeds[0])
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
kmeans_model = kmeans_algo1.fit(X)
y_pred = kmeans_model.labels_
y_true = y
print("Silhouette:",metrics.silhouette_score(X, y_pred, metric='euclidean'))
print("Purity:",purity_score(y_true, y_pred))
"OLA"

kmeans_algo2 = cluster.KMeans(n_clusters=3, random_state=seeds[1])
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
kmeans_model = kmeans_algo2.fit(X)
y_pred = kmeans_model.labels_
print("Silhouette:",metrics.silhouette_score(X, y_pred, metric='euclidean'))
print("Purity:",purity_score(y_true, y_pred))


kmeans_algo3 = cluster.KMeans(n_clusters=3, random_state=seeds[2])
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
kmeans_model = kmeans_algo3.fit(X)
y_pred = kmeans_model.labels_
print("Silhouette:",metrics.silhouette_score(X, y_pred, metric='euclidean'))
print("Purity:",purity_score(y_true, y_pred))


