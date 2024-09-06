import numpy as np

def hamming(X, x):
    print("x = ", x)

    w = X / 2

    print("w = ", w)

    b = X.shape[1] / 2 * np.ones(X.shape[0])

    print("b = ", b)

    u = (np.dot(w, x) + b) / w.shape[1]

    # If u is smaller than 0, set it to 0, bigger than 1, set it to 1, otherwise set it to u
    u = np.where(u < 0, 0, u)
    u = np.where(u > 1, 1, u)

    print("u = ", u)

    new = np.ones(u.shape[0])
    epoch = 0
    # While new has more than one element different from zero
    while np.count_nonzero(new) > 1:
        epoch += 1
        for i in range(u.shape[0]):
            # Sum all except the i-th
            new[i] = u[i] - (1 / (w.shape[1] - 1)) * (np.sum(np.delete(u, i)))

        new = np.where(new < 0, 0, new)
        new = np.where(new > 1, 1, new)

        u = new
        print("epoch ", epoch, " = ", new)

    # return the index of the max element
    return np.argmax(new)

X = np.array([[1, 1, 1], [-1, -1, -1], [1, -1, 1]])
x = np.array([-1, 1, 1])
print(hamming(X, x))