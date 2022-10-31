from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
from sklearn import metrics, datasets, tree
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler

data = loadarff('pd_speech.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].str.decode('utf-8')
X = df.drop('class', axis=1)
y = df['class']

cumulative = np.zeros((2, 2), dtype=int)
acc_knn = []
acc_nb = []
folds = StratifiedKFold(n_splits=10,random_state=0, shuffle = True)
predictor = KNeighborsClassifier(n_neighbors=5,weights='uniform',p=2)

for train_k, test_k in folds.split(X, y):
    X_train, X_test = X.iloc[train_k], X.iloc[test_k]
    y_train, y_test = y.iloc[train_k], y.iloc[test_k]
    scaler = StandardScaler()
    X_train, X_test =  scaler.fit_transform(X_train), scaler.fit_transform(X_test)

    predictor.fit(X_train, y_train)
    y_pred = predictor.predict(X_test)
    acc_knn.append(round(metrics.accuracy_score(y_test, y_pred),5))
    cm = np.array(confusion_matrix(y_test, y_pred))
    cumulative = np.add(cumulative, cm)
    
second_cumulative = np.zeros((2, 2), dtype=int)
predictor = GaussianNB()

for train_k, test_k in folds.split(X, y):
  
    X_train, X_test = X.iloc[train_k], X.iloc[test_k]
    y_train, y_test = y.iloc[train_k], y.iloc[test_k]
    scaler = StandardScaler()
    X_train, X_test =  scaler.fit_transform(X_train), scaler.fit_transform(X_test)

    predictor.fit(X_train, y_train)
    y_pred = predictor.predict(X_test)
    acc_nb.append(round(metrics.accuracy_score(y_test, y_pred),5))
    cm = np.array(confusion_matrix(y_test, y_pred))
    second_cumulative = np.add(second_cumulative, cm)

print("kNN Accuracies:", acc_knn)
print("NB Accuracies:", acc_nb)

fig, ax =plt.subplots(2,1)
plt.subplots_adjust(hspace = 0.5)

confusion = pd.DataFrame(cumulative,index=['Positive', 'Negative'], columns=['Predicted Positive', 'Predicted Negative'])
kNN = sns.heatmap(confusion, annot=True,fmt='g', ax=ax[0])
kNN.set(title='kNN')

confusion_2 = pd.DataFrame(second_cumulative,index=['Positive', 'Negative'], columns=['Predicted Positive', 'Predicted Negative'])
NB = sns.heatmap(confusion_2, annot=True, fmt='g', ax=ax[1])
NB.set(title='Naive Bayes')

res = stats.ttest_rel(acc_knn, acc_nb, alternative='greater')
print("p1>p2? pval=",res.pvalue)