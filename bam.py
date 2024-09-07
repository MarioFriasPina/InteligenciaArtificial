import numpy as np

x1 = np.array([1, 1, 1])
y1 = np.array([-1, -1, -1, -1, -1])

x2 = np.array([1, -1, 1])
y2 = np.array([-1, 1, -1, 1, -1])

# Multiply every element of x1 with every element of y1
# and every element of x2 with every element of y2
# and add them together

w = np.zeros((x1.shape[0], y1.shape[0]))

for i in range(w.shape[0]):
    for j in range(w.shape[1]):
        w[i][j] = (x1[i] * y1[j] + x2[i] * y2[j])

#w = np.array([[-2, 0, -2], [0, -2, 0], [-2, 0, -2]])

yp1 = [1, 1, 1, 1, 1]
yp2 = [-1, -1, -1, -1, 1]
yp3 = [-1, 1, -1, 1, 1]

print(w)

#res = np.dot(y, w.T)
#res = np.dot(x, w)

print("yp1")
res = np.dot(yp1, w.T)
print(res)
print(np.where(res > 0 , 1, -1))

print("yp2")
res = np.dot(yp2, w.T)
print(res)
print(np.where(res > 0 , 1, -1))

print("yp3")
res = np.dot(yp3, w.T)
print(res)
print(np.where(res > 0 , 1, -1))