import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB #usata per l'antispam
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

twitter_dataset = pd.read_csv('data.csv', delimiter = ',', header = 0)

X = twitter_dataset[['text']]
y = twitter_dataset[['sentiment']]

encoder = OneHotEncoder(handle_unknown='ignore')
X = encoder.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel())

model = BernoulliNB()
model.fit(X_train, y_train)

train_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)

print('Train_Prediction: ', train_prediction)
print('Test_Prediction: ', test_prediction)

train_accuracy = accuracy_score(y_train, train_prediction)
test_accuracy = accuracy_score(y_test, test_prediction)

print('Accuracy_train: ', train_accuracy * 100)
print('Accuracy_test: ', test_accuracy * 100)



fig, axes = plt.subplots()
axes.plot(train_accuracy * 100, '.', color = 'red', label = 'Train')
axes.plot(test_accuracy * 100, '.', color = 'blue', label = 'Test')
axes.legend()
plt.show()