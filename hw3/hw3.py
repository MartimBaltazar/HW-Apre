from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
import seaborn as sns

graph1 = []
graph2 = []
graph3 = []
fig, axs = plt.subplots(2,3)

data = loadarff('kin8nm.arff')
df = pd.DataFrame(data[0])
X = df.drop('y', axis=1)
print(X.dtypes)
y = df['y']
y=y.astype('float')

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,random_state=0)

lr = Ridge(alpha=0.1)
lr.fit(X_train, y_train)
pred_test_lr= lr.predict(X_test)
y_true, y_pred = y_test, pred_test_lr
print("MAE RIDGE:",metrics.mean_absolute_error(y_true, y_pred))
for i in  range(len(y_true)):
    value = abs(y_true.array[i]-y_pred[i])
    graph1.append(value)

MLP1 = MLPRegressor(random_state=0, hidden_layer_sizes=(10,10), activation='tanh', max_iter=500,early_stopping=True)
var = MLP1.fit(X_train, y_train)
pred_test_MLP1= MLP1.predict(X_test)
y_true, y_pred = y_test, pred_test_MLP1
print("MAE MLP1:",metrics.mean_absolute_error(y_true, y_pred))
for i in  range(len(y_true)):
    value = abs(y_true.array[i]-y_pred[i])
    graph2.append(value)

MLP2 = MLPRegressor(random_state=0, hidden_layer_sizes=(10,10), activation='tanh', max_iter=500,early_stopping=False)
MLP2.fit(X_train, y_train)
pred_test_MLP2= MLP2.predict(X_test)
y_true, y_pred = y_test, pred_test_MLP2
print("MAE MLP2:",metrics.mean_absolute_error(y_true, y_pred))
for i in  range(len(y_true)):
    value = abs(y_true.array[i]-y_pred[i])
    graph3.append(value)

fig.tight_layout()
axs[0][0].set_title('Ridge')
axs[0][0].hist(graph1)
axs[1][0].boxplot(graph1)
axs[0][1].set_title('MLP1')
axs[0][1].hist(graph2)
axs[1][1].boxplot(graph2)
axs[0][2].set_title('MLP2')
axs[0][2].hist(graph3)
axs[1][2].boxplot(graph3)

print(MLP1.n_iter_,MLP2.n_iter_)