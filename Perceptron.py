import numpy as np
import matplotlib.pyplot as plt

table = np.array([[1, 0, 0, 0], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]])

def step(X):
    return np.where(X > 0, 1, 0)

def perceptron(X, y, alpha = 0.1, iteraciones = 1500):
    # Initialize weights with random values between 0 and 1
    w = np.random.rand(X.shape[1])
    for i in range(iteraciones):
        # Predict
        out = step(np.dot(X, w))
        # Calculate new weights
        w_t = w + alpha * X.T.dot(y - out)

        print("Current weights: ", w)
        print("Predicted output: ", out)
        print("Actual output: ", y)

        # Check for convergence
        if np.array_equal(w, w_t):
            print('Converged after ' + str(i) + ' iterations')
            break

        # Update
        w = w_t

    return w

def plot(X, y, w):

    plt.figure()

    plt.scatter(X[:, 1], X[:, 2], c=y)

    # Plot decision boundary
    x = np.linspace(-0.5, 1.5, 100)
    y = -(w[0] + w[1] * x) / w[2]
    plt.plot(x, y, '-r')

    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)

    plt.show()

predicted = perceptron(table[:, 0:3], table[:, -1])

plot(table[:, 0:3], table[:, -1], predicted)