from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, metrics, cluster, mixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def purity_score(y_true, y_pred):
    confusion_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix) 

data = loadarff('pd_speech.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].str.decode('utf-8')
X = df.drop('class', axis=1)
y = df['class']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
plt.tight_layout()

first_variable = X.iloc[:, 0].var()
second_variable = first_variable
first_array = X.iloc[:, 0]
second_array = first_array
        
label = y.to_numpy()
color = list(map(lambda x : 'red' if x=="1" else 'blue',label))

for i in range(3):
    kmeans_algo1 = cluster.KMeans(n_clusters=3, random_state=i)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    kmeans_model = kmeans_algo1.fit(X)
    y_pred = kmeans_model.labels_
    y_true = y
    print("Silhouette:",metrics.silhouette_score(X, y_pred, metric='euclidean'))
    print("Purity:",purity_score(y_true, y_pred))
    X = pd.DataFrame(X)
    if i == 0:
        for column in X:
            if X[column].var() > first_variable:
                second_variable = first_variable
                first_variable = X[column].var()
                second_array = first_array
                first_array = X[column]
            elif X[column].var() > second_variable:
                second_variable = X[column].var()
                second_array = X[column]
        ax1.set_title("Parkinson")
        ax1.scatter(first_array,second_array,c=color)
        ax2.set_title("Kmeans Clustering")
        ax2.scatter(first_array,second_array,c=kmeans_algo1.labels_)

pca = PCA(n_components=752)
pca.fit(X)

total = sum(pca.explained_variance_ratio_)

percentage = 0
necessary_components = 0

for i in range(752):
    percentage += pca.explained_variance_ratio_[i] / total
    necessary_components+=1
    if percentage >= 0.8:
        break
print("necessary_components: ",necessary_components)