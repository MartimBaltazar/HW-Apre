data = loadarff('pd_speech.arff')
df = pd.DataFrame(data[0])
print(df.head())

x, y = df.drop("class", axis=1), np.ravel(df['class'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=1)
print("train size:",len(x_train),"\ntest size:",len(x_test))

from sklearn import metrics
print("accuracy:",  round(metrics.accuracy_score(y_test, y_train),2))
print("recall/sensitivity:", round(metrics.recall_score(y_test, y_train),2))
print("precision:", round(metrics.precision_score(y_test, y_train),2))
