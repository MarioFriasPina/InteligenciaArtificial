import pandas as pd
import numpy as np

import plotly.express as px

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def stochastic_gradient_descent(X, y, alpha = 0.01, iters = 1000, activation = sigmoid):

    # Initialize weights with random values between 0 and 1
    weights = np.random.rand(X.shape[1])

    for _ in range(iters):
        # For each training example
        for i in range(X.shape[0]):
            # Calculate a new alpha each iteration that gets smaller as we go through the dataset
            a = alpha * 4 / (1 + i + _)

            # Select a random training example
            random_index = np.random.randint(X.shape[0])

            # Calculate the gradient for this training example
            gradient = activation(X[random_index].dot(weights))

            # Update the weights
            weights = weights - a * (gradient - y[random_index]) * X[random_index]

    return weights

def predict(X, weights, activation = sigmoid):
    return np.round(activation(X.dot(weights)))

def get_weights_different_parameters(X, y):
    weights = {}
    # For each activation function, learning rate and number of iterations
    for activation in [sigmoid, tanh, relu]:
        for alpha in [0.01, 0.03, 0.1, 0.3]:
            for iters in [1, 10, 50, 100, 150]:
                weights[activation][alpha][iters] = stochastic_gradient_descent(X, y, alpha, iters, activation)

    return weights

def __main__():
    # Load the data
    training = pd.read_csv('LogisticRegression/cancerTraining.txt')
    test = pd.read_csv('LogisticRegression/cancerTest.txt')

    # Obtain the labels from the training and test sets
    train_labels = training.iloc[:, -1]
    test_labels = test.iloc[:, -1]

    # Remove the labels from the training and test sets
    training = training.drop(training.columns[-1], axis=1)
    test = test.drop(test.columns[-1], axis=1)

    # Obtain the weights
    #weights = stochastic_gradient_descent(training.values, train_labels.values, 0.01, 1000)

    #predicted_labels = predict(test.values, weights)

    # Use scikit-learn

    log = LogisticRegression(max_iter=1000, tol=0.01)

    log.fit(training.values, train_labels.values)

    predicted_labels_sk = log.predict(test.values)

    # Calculate accuracy
    #print(accuracy_score(test_labels.values, predicted_labels))
    print(accuracy_score(test_labels.values, predicted_labels_sk))

__main__()