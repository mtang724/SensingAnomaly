import pprint, pickle
# Sklearn regression algorithms
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
# Sklearn regression model evaluation function
from sklearn.metrics import mean_absolute_error

from sklearn.neural_network import MLPRegressor
import numpy as np

X_file = open('datafileX.txt', 'rb')

dataX = pickle.load(X_file)
print(dataX.shape)

X_file.close()

y_file = open('datafileY.txt', 'rb')

dataY = pickle.load(y_file)
pprint.pprint(dataY)

y_file.close()
# print(dataX.tolist())
nsamples, nx, ny = dataX.shape
dataX = dataX.reshape((nsamples,nx*ny))
X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=42)

models = [LinearRegression(), KNeighborsRegressor(), SVR(), DecisionTreeRegressor()]
for model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    print(type(model).__name__, mean_absolute_error(y_train, predictions))

for model in models:
    predictions = model.predict(X_test)
    print(type(model).__name__, mean_absolute_error(y_test, predictions))