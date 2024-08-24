import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def graficaDatos(X, y, theta):
    """
    Grafica la recta de regresion y la recta de decision

    Argumentos:
        X (vector): El vector de variables de entrada
        y (vector): El vector de variables de salida reales
        theta (vector): El vector de parámetros calculados

    """
    plt.figure()
    plt.title('Regresion Lineal')

    #Obtener los indices de los valores de 1 y 0
    idx0 = np.where(y == 1)
    idx1 = np.where(y == 0)

    plt.scatter(X[idx0, 1], X[idx0, 2], marker='x', color='red', label='No admitido')
    plt.scatter(X[idx1, 1], X[idx1, 2], marker='o', color='green', label='Admitido')

    plt.xlabel('Examen 1 (Normalizado)')
    plt.ylabel('Examen 2 (Normalizado)')

    plt.legend(loc='best')

    x = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)

    # Imprimir la recta de decision si el vector de parámetros tiene tamaño 2
    if theta.size == 2:
        # y = mx + b
        y = -(theta[0] + theta[1] * x) / theta[2]
    else:
        # y = mx^2 + mx + b
        y = -(theta[0] + theta[1] * x + theta[3] * x**2) / theta[2]
    
    plt.plot(x, y, '-b')
    plt.xlim(min(x), max(x))
    plt.ylim(min(y), max(y))

    plt.show()

def sigmoidal(z):
    """
    Calcula la sigmoidal de z

    sigmoidal = 1 / (1 + e^(-z))

    Argumentos:
        z (float): El valor de z
    """
    return 1 / (1 + np.exp(-z))

def funcionCosto(theta, X, y):
    """
    Calcula la función de costo para el modelo de regresión logística

    J(theta) = 1/m * sum(-y * log(h(x)) - (1 - y) * log(1 - h(x)))

    Argumentos:
        theta (vector): El vector de los pesos
        X (vector): El vector de variables de entrada
        y (vector): El vector de variables de salida reales

    Regreso:
        float: El valor de la función de costo
        vector: El vector de gradiente
    """
    h = sigmoidal(X.dot(theta))

    # Ajuste numérico para evitar log(0)
    # https://stackoverflow.com/questions/21610198/runtimewarning-divide-by-zero-encountered-in-log
    J = (-y * np.log(h, where=h > 0) - (1 - y) * np.log(1 - h, where=h < 1)).sum() / len(y)
    grad = X.T.dot(h - y) / len(y)

    return [J, grad]

def aprende(theta, X, y, iteraciones, a = 0.1):
    """
    Entrena el modelo de regresión logística

    Argumentos:
        theta (vector): El vector de los pesos
        X (vector): El vector de variables de entrada
        y (vector): El vector de variables de salida reales
        iteraciones (int): El número de iteraciones maximas
        a (float): El valor de la tasa de aprendizaje

    Regreso:
        vector: El vector de pesos calculados
    """
    for _ in range(iteraciones):
        # Calcula el gradiente con la formula de hipotesis lineal
        h = sigmoidal(X.dot(theta))
        gradient = X.T.dot(h - y) / len(y)

        # Actualiza los pesos
        theta = theta - a * gradient

    return theta

def predice(theta, X):
    """
    Realiza una predicción con el modelo de regresión logística

    Argumentos:
        theta (vector): El vector de los pesos
        X (vector): El vector de variables de entrada

    Regreso:
        float: El valor de la variable de salida
    """
    # Si la entrada es un vector
    if X.ndim == 1:
        return 1 if sigmoidal(X.dot(theta)) >= 0.5 else 0
    
    return [1 if i >= 0.5 else 0 for i in sigmoidal(X.dot(theta))]

# WARNING: No utilizar la librería de pandas para leer el archivo ya que es mucho mas lento
#X = pd.read_csv('Proyecto2/ex2data1.txt')
#y = X.iloc[:, -1]
#X = X.drop(X.columns[-1], axis=1)

# Leer el archivo de texto con las variables x e y
X = np.loadtxt('Proyecto2/ex2data1.txt', delimiter=',')
y = X[:, -1]
X = X[:, 0:-1]

# Obtener la media y la desviación estándar de X
mean = [np.mean(X[:, 0]), np.mean(X[:, 1])]
std = [np.std(X[:, 0]), np.std(X[:, 1])]

# Normalizar X
X[:, 0] = (X[:, 0] - mean[0]) / std[0]
X[:, 1] = (X[:, 1] - mean[1]) / std[1]

# Agregar dos columnas de cuadrados
X = np.c_[X, X[:, 0] ** 2]

# Agregar una columna de 1s
X = np.c_[np.ones(X.shape[0]), X]

# Iniciar los pesos aleatorios
theta = np.random.rand(X.shape[1])

w = aprende(theta, X, y, 200000, 0.01)

print("Costo final: ", funcionCosto(w, X, y)[0])

# Predecir la probabilidad de ser admitido para un estudiante con 45 y 85
x = np.array([1, (45 - mean[0]) / std[0], (85 - mean[1]) / std[1]])

#print("Probabilidad de ser admitido para un estudiante con 45 y 85: {:.3f} por lo que predice {} ".format( sigmoidal(np.dot(x, w)), predice(w, x)))

# Graficar la recta de decision
graficaDatos(X, y, w)