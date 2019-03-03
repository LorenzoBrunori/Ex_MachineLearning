import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB #usata per l'antispam
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

twitter_dataset = pd.read_csv('data.csv', delimiter = ',', header = 0)

X = twitter_dataset[['text']]
y = twitter_dataset[['sentiment']]

encoder = OneHotEncoder(handle_unknown='ignore')
X = encoder.fit_transform(X)

temp_y = y.values.ravel()

model = BernoulliNB()
model.fit(X, temp_y)

prediction = model.predict(X)
print('Prediction: ', prediction)

accuracy = accuracy_score(temp_y, prediction)
print('Accuracy: ', accuracy)