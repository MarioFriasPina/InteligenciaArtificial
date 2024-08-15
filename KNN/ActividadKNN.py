import numpy as np
import pandas as pd
import math
from operator import itemgetter
from statistics import mode
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def euclidean_distance(vec1, vec2):
    """
    Euclidean distance is the square root of the sum of the squared differences between the elements of the two vectors.
    """
    distance = 0.0
    for i in range(len(vec1)):
        distance += (vec1.iloc[i] - vec2.iloc[i]) ** 2
    return math.sqrt(distance)

def manhattan_distance(vec1, vec2):
    """
    Manhattan distance is the sum of the absolute differences between the elements of the two vectors.
    """
    distance = 0.0
    for i in range(len(vec1)):
        distance += abs(vec1.iloc[i] - vec2.iloc[i])
    return distance

def chebyshev_distance(vec1, vec2):
    """
    Chebychev distance is the maximum of the absolute differences between the elements of the two vectors.
    """
    distance = 0.0
    for i in range(len(vec1)):
        distance = max(distance, abs(vec1.iloc[i] - vec2.iloc[i]))
    return distance

def classify(testList, trainLists, trainLabels, k, metric):
    """
    Classify the test point based on the k nearest neighbors in the training set.

    Args:
        testList (list): The test point.
        trainLists (list): The training set.
        trainLabels (list): The labels of the training set.
        k (int): The number of nearest neighbors to consider.
        metric (function): The distance metric to use.

    Returns:
        str: The predicted label.
    """
    distance = []

    for i in range(len(trainLists)):
        # Get the distance between the test point and the current training point
        dist = metric(testList, trainLists.iloc[i])
        #dist = chebyshev_distance(testList, trainLists.iloc[i])
        distance.append((dist, trainLabels.iloc[i]))

    distance.sort(key=itemgetter(0))

    voteLabels = [i[1] for i in distance[:k]]

    return mode(voteLabels)

def calculate_accuracy(test_labels, predicted_labels):
    """
    Calculate the accuracy of the predictions.

    Args:
        test_labels (list): The true labels of the test set.
        predicted_labels (list): The predicted labels of the test set.

    Returns:
        float: The accuracy of the predictions.
    """
    correct = 0
    for i in range(len(test_labels)):
        if test_labels[i] == predicted_labels[i]:
            correct += 1
    return correct / float(len(test_labels)) * 100.0

def train_knn(train, train_labels, test, test_labels, k, metric = manhattan_distance):
    """
    Train the KNN model and calculate the accuracy of the predictions.
    
    Args:
        train (DataFrame): The training set.
        train_labels (list): The labels of the training set.
        test (DataFrame): The test set.
        test_labels (list): The labels of the test set.
        k (int): The number of nearest neighbors to consider.
        metric (function): The distance metric to use.

    Returns:
        list : The predicted labels of the test set.
    """
    predicted_labels = []

    for i in range(len(test)):
        predicted_labels.append(classify(test.iloc[i, :], train, train_labels, k, metric))

    return predicted_labels
    #return calculate_accuracy(test_labels, predicted_labels)

def get_best_k(training, train_labels, test, test_labels, range_list):
    """
    Calculate the accuracy of the predictions for different values of k.

    Args:
        training (DataFrame): The training set.
        train_labels (list): The labels of the training set.
        test (DataFrame): The test set.
        test_labels (list): The labels of the test set.
        range_list (list): The list of values of k to consider.

    Returns:
        accuracies (list): The accuracy of the predictions for each value of k and distance metric.
        predicted (list): The predicted labels of the test set for each value of k and distance metric.
    """
    accuracies = {euclidean_distance: [], manhattan_distance: [], chebyshev_distance: []}
    predicted = {euclidean_distance: [], manhattan_distance: [], chebyshev_distance: []}

    for metric in [euclidean_distance, manhattan_distance, chebyshev_distance]:
        for k in range_list:
            predicted_labels = train_knn(training, train_labels, test, test_labels, k, metric)
            accuracy = calculate_accuracy(test_labels, predicted_labels)
            predicted[metric].append(predicted_labels)
            accuracies[metric].append(accuracy)
            print(f'k = {k}, accuracy = {accuracy}, distance = {metric.__name__}')
    return accuracies, predicted

def plot_best_k(range_list, accuracies):
    """
    Plot the accuracy of the predictions for different values of k.

    Args:
        range_list (list): The list of values of k to consider.
        accuracies (list): The accuracy of the predictions for each value of k.

    Returns:
        list: The best value of k for each distance metric.
    """

    plt.figure()
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Best K")
    plt.xticks(range_list)

    for metric in [euclidean_distance, manhattan_distance, chebyshev_distance]:
        plt.plot(range_list, accuracies[metric], label = metric.__name__)

    plt.legend()

    best_k = [accuracies[metric].index(max(accuracies[metric])) for metric in [euclidean_distance, manhattan_distance, chebyshev_distance]]
    return best_k

def compare_metrics(accuracies, best_k):
    """
    Compare the accuracy of the predictions for different distance metrics.

    Args:
        accuracies (list): The accuracy of the predictions for each value of k and distance metric.
        best_k (list): The best value of k for each distance metric.

    Returns:
        None
    """

    plt.figure()
    plt.bar([euclidean_distance.__name__, manhattan_distance.__name__, chebyshev_distance.__name__],
            [accuracies[euclidean_distance][best_k[0]],
             accuracies[manhattan_distance][best_k[1]],
             accuracies[chebyshev_distance][best_k[2]]])
    plt.title("Accuracy by Distance Metric")
    plt.xlabel("Metric")
    plt.ylabel("Accuracy")

# Other plots that may be useful

def plot_confusion_matrix(test, predicted, best_k):
    """
    Plot the confusion matrix of the test set.

    Args:
        test (list): The test labels of the test set.
        predicted (list): The predicted labels of the test set.
        best_k (list): The best value of k for each distance metric.

    Returns:
        None
    """

    i = 0
    for metric in [euclidean_distance, manhattan_distance, chebyshev_distance]:
        cm = confusion_matrix(test, predicted[metric][best_k[i]])
        disp = ConfusionMatrixDisplay(cm, display_labels = test.unique())
        disp.plot()
        plt.title ("Confusion Matrix for {} metric with k = {}".format(metric.__name__, best_k[i] + 3))
        i += 1

def plot_correlation_matrix(train):
    """
    Plot the correlation matrix of the training set.

    Args:
        train (DataFrame): The training set.

    Returns:
        None
    """
    plt.figure()
    plt.title("Correlation Matrix")
    plt.imshow(train.corr())
    plt.colorbar()

def plot(range_list, accuracies, training, test_labels, predicted_labels):
    """
    Plot all data

    Args:
        range_list (list): The list of values of k to consider.
        accuracies (list): The accuracy of the predictions for each value of k and distance metric.
        training (DataFrame): The training set.
        test_labels (list): The labels of the test set.
        predicted_labels (list): The predicted labels of the test set.

    Returns:
        None
    """

    # Obtain the best value of k and plot the accuracy
    best_k = plot_best_k(range_list, accuracies)

    # Plot the correlation matrix
    plot_correlation_matrix(training)

    # Plot the confusion matrix
    plot_confusion_matrix(test_labels, predicted_labels, best_k)

    # Compare the accuracy of the predictions for different distance metrics
    compare_metrics(accuracies, best_k)

def __main__():
    # Load the data
    training = pd.read_csv('KNN/training.txt')
    test = pd.read_csv('KNN/test.txt')

    # Obtain the labels from the training and test sets
    train_labels = training.iloc[:, -1]
    test_labels = test.iloc[:, -1]

    # Remove the labels from the training and test sets
    training = training.drop(training.columns[-1], axis=1)
    test = test.drop(test.columns[-1], axis=1)

    # Range of k values to consider
    range_list = range(3, 42)

    # Obtain the accuracy for different values of k for each distance metric
    accuracies, predicted_labels = get_best_k(training, train_labels, test, test_labels, range_list)

    # Utilize KNeighborsClassifier from scikitlearn
    sk_accuracies = {euclidean_distance: [], manhattan_distance: [], chebyshev_distance: []}
    sk_predicted_labels = {euclidean_distance: [], manhattan_distance: [], chebyshev_distance: []}
    for metric in [euclidean_distance, manhattan_distance, chebyshev_distance]:
        for k in range_list:
            neigh = KNeighborsClassifier(n_neighbors=k, metric=metric.__name__.split('_')[0] if metric.__name__ != 'chebyshev_distance' else 'cosine')
            neigh.fit(training.values, train_labels.values)
            sk_predicted_labels[metric].append(neigh.predict(test.values))
            sk_accuracies[metric].append(accuracy_score(test_labels.values, sk_predicted_labels[metric][-1]))
            #sk_accuracies.append(calculate_accuracy(test_labels, sk_predicted_labels))
            #print(f'k = {k}, accuracy = {sk_accuracies[metric][-1]}, distance = {metric.__name__}')

    plot(range_list, sk_accuracies, training, test_labels, sk_predicted_labels)
    plot(range_list, accuracies, training, test_labels, predicted_labels)

    plt.show()

__main__()