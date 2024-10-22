import numpy as np
def per(x, y, lr=0.5):
iter = 4
weights = np.array([0.9, 0.9])
bias = 0
for _ in range(iter):
for i in range(len(x)):
output = np.dot(x[i], weights) + bias
if output >= 0.5:
predicted = 1
else :
predicted=0
update = lr * (y[i] - predicted)
weights += update * x[i]
bias += update
return weights, bias
def predict(x, weights, bias):
return 1 if np.dot(x, weights) + bias >= 0.5 else 0
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
weights, bias = per(X_and, y_and)
predictions = np.array([predict(x, weights, bias) for x in X_and])
print("AND Predictions:", predictions)

import numpy as np
def per(x, y, lr=0.5):
iter = 4
weights = np.array([0.9, 0.9])
bias = 0
for _ in range(iter):
for i in range(len(x)):
output = np.dot(x[i], weights) + bias
if output >= 0.5:
predicted = 1
else :
predicted=0
update = lr * (y[i] - predicted)
weights += update * x[i]
bias += update
return weights, bias
def predict(x, weights, bias):
return 1 if np.dot(x, weights) + bias >= 0.5 else 0
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])
weights, bias = per(X_or, y_or)
predictions = np.array([predict(x, weights, bias) for x in X_and])
print(" OR Predictions:", predictions)

import numpy as np
def per(x, y, lr=0.5):
iter = 100
weights = np.array([0.9, 0.9])
bias = 0
for _ in range(iter):
for i in range(len(x)):
output = np.dot(x[i], weights) + bias
if output >= 0.5:
predicted = 1
else :
predicted=0
update = lr * (y[i] - predicted)
weights += update * x[i]
bias += update
return weights, bias
def predict(x, weights, bias):
return 1 if np.dot(x, weights) + bias >= 0.5 else 0
X_or = np.array([[1, 2], [2, 3], [3, 1], [4, 3]])
y_or = np.array([0, 0, 1, 1])
weights, bias = per(X_or, y_or)
predictions = np.array([predict(x, weights, bias) for x in X_and])
print("Predictions:", predictions)
