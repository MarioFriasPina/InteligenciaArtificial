import numpy as np
import matplotlib.pyplot as plt

import random as rand

OR_table = np.array([[1, 0, 0, 0], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])

initial_weigths = [1.5, 0.5, 1.5]

def escalon(x):
    return 1 if x >= 0 else 0

def perceptron(X, y, theta, alpha = 0.01, iteraciones = 1500):
    for i in range(iteraciones):
        for j in range(X.shape[0]):
            d = X[j].dot(theta)
            #y_pred = escalon(d)

            theta = theta + alpha * (y[j] - d) * X[j]

    print(theta)
    return theta

def plot(X, y, theta):

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.plot(X, X*theta[1] + theta[0], color='red')
    plt.show()

predicted = perceptron(OR_table[:, 0:3], OR_table[:, -1], initial_weigths)

plot(OR_table[:, 1:3], OR_table[:, -1], predicted)