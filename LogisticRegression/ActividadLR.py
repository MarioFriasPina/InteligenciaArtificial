import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.utils.discovery import all_displays
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import warnings

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):    
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def stochastic_gradient_descent(X, y, alpha = 0.01, iters = 1000, activation = sigmoid):
    # Initialize weights with random values between 0 and 1
    weights = np.random.rand(X.shape[1])

    list_weights = []
    for _ in range(iters):
        # For each training example
        for i in range(X.shape[0]):
            # Calculate a new alpha each iteration that gets smaller as we go through the dataset
            a = 4 / (1 + i + _) + alpha

            # Select a random training example
            random_index = np.random.randint(X.shape[0])

            # Calculate the gradient for this training example
            gradient = activation(X[random_index].dot(weights))

            # Update the weights
            weights = weights - a * (gradient - y[random_index]) * X[random_index]

        list_weights.append(weights)

    return np.array(list_weights)

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

def plot(manual_results, scikit_results, ):
    # PLOTLY
    fig = px.scatter(title=f"Precision and Recall from manual implementation", range_y=[0, 1])

    for alpha in [0.01, 0.03, 0.1, 0.3]:
        fig.add_scatter(x=np.array(range(len(manual_results["precision"][alpha]))), y=manual_results["precision"][alpha], name=f"Precision with alpha = {alpha}")

    for alpha in [0.01, 0.03, 0.1, 0.3]:
        fig.add_scatter(x=np.array(range(len(manual_results["recall"][alpha]))), y=manual_results["recall"][alpha], name=f"Recall with alpha = {alpha}")

    fig.show()

    fig = px.scatter(title=f"Precision and Recall from scikit-learn implementation", range_y=[0, 1])

    for alpha in [0.01, 0.03, 0.1, 0.3]:
        fig.add_scatter(x=np.array(range(len(scikit_results["precision"][alpha]))), y=scikit_results["precision"][alpha], name=f"Precision with alpha = {alpha}")

    for alpha in [0.01, 0.03, 0.1, 0.3]:
        fig.add_scatter(x=np.array(range(len(scikit_results["recall"][alpha]))), y=scikit_results["recall"][alpha], name=f"Recall with alpha = {alpha}")

    fig.show()

def plot_feature_importance(weights):
    parameters = ("Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses")

    fig = px.bar(x=parameters, y=weights)
    fig.show()

def __main__():
    #Remove warnings
    warnings.filterwarnings("ignore")

    # Load the data
    training = np.loadtxt('LogisticRegression/cancerTraining.txt', delimiter=',')
    test = np.loadtxt('LogisticRegression/cancerTest.txt', delimiter=',')

    # Obtain the labels from the training and test sets
    train_labels = training[:, -1]
    test_labels = test[:, -1]

    # Remove the labels from the training and test sets
    training = training[:, :-1]
    test = test[:, :-1]

    results_manual = {"precision": { 0.01: [], 0.03: [], 0.1: [], 0.3: []}, "recall": { 0.01: [], 0.03: [], 0.1: [], 0.3: []}}
    results_sklearn = {"precision": { 0.01: [], 0.03: [], 0.1: [], 0.3: []}, "recall": { 0.01: [], 0.03: [], 0.1: [], 0.3: []}}

    all_predictions = {"manual": [], "sklearn": [], "real": []}
    for alpha in [0.01, 0.03, 0.1, 0.3]:
        # Obtain the weights manually using stochastic gradient descent
        weights = stochastic_gradient_descent(training, train_labels, alpha, 151)

        for iters in range(1, 151):
            predicted_labels = predict(test, weights[iters])

            # Save the results
            results_manual["precision"][alpha].append(precision_score(test_labels, predicted_labels))
            results_manual["recall"][alpha].append(recall_score(test_labels, predicted_labels))

            # Use scikit-learn's Logistic Regression
            clf = LogisticRegression(max_iter = iters, verbose=0, C = alpha).fit(training, train_labels)
            predicted_labels_sklearn = clf.predict(test)

            all_predictions["manual"].extend(predicted_labels)
            all_predictions["sklearn"].extend(predicted_labels_sklearn)
            all_predictions["real"].extend(test_labels)

            # Save the results
            results_sklearn["precision"][alpha].append(precision_score(test_labels, predicted_labels_sklearn))
            results_sklearn["recall"][alpha].append(recall_score(test_labels, predicted_labels_sklearn))

    # Plot precision and recall through iterations
    plot(results_manual, results_sklearn)

    # Plot the most important features
    plot_feature_importance(weights[-1])

    # Confusion matrix average results
    ConfusionMatrixDisplay(confusion_matrix(all_predictions["real"], all_predictions["manual"], normalize="true")).plot()
    ConfusionMatrixDisplay(confusion_matrix(all_predictions["real"], all_predictions["sklearn"], normalize="true")).plot()

    plt.show()

__main__()