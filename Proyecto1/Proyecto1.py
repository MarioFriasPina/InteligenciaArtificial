import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def graficaDatos(X, y, theta, costs = None):
    """
    Grafica la recta de regresion y la recta de costos

    Argumentos:
        X (vector): El vector de variables de entrada
        y (vector): El vector de variables de salida reales
        theta (vector): El vector de parámetros calculados
        costs (vector): El vector de costos calculados

    """
    # Graficar la recta de costos si costs no es None
    if costs is not None:
        plt.figure()
        plt.title('Costos')
        plt.xlabel('Iteraciones')
        plt.ylabel('Costo promedio')
        plt.plot(costs)

    # Graficar la recta de regresion
    plt.figure()
    plt.title('Regresion Lineal')
    plt.scatter(X, y)
    plt.plot(X, X*theta[1] + theta[0], color='red')
    plt.show()

def gradienteDescendiente(X, y, theta = [0, 0], alpha = 0.01, iteraciones = 1500):
    """
    Calcula el gradiente descendiente para el modelo de regresión lineal

    Theta[0] = Theta[0] - alpha * (1/m) * sum(X * (h(x) - y))
    Theta[1] = Theta[1] - alpha * (1/m) * sum((h(x) - y) * X)

    h(x) = theta[0] + theta[1] * x = X * theta

    Argumentos:
        X (vector): El vector de variables de entrada 
        y (vector): El vector de variables de salida reales
        theta (vector): El vector de parámetros iniciales
        alpha (float): El valor de la tasa de aprendizaje
        iteraciones (int): El número de iteraciones maximas

    Regreso:
        vector: El vector de parámetros calculados
    """
    costs = []
    for i in range(iteraciones):
        # Calcular el gradiente con la formula de hipotesis lineal
        t0 = theta[0] - alpha *  (1/len(X)) * np.sum((X.dot(theta) - y))
        t1 = theta[1] - alpha * (1/len(X)) * np.sum((X.dot(theta) - y) * X[:,1]) # X[:,1] es la segunda columna de X que es la variable x
        

        # Terminar cuando theta no cambie
        if theta[0] == t0 and theta[1] == t1:
            break
        theta = [t0, t1]

        # Calcular el costo en cada iteracion
        costs.append(calculaCosto(X, y, theta))

    graficaDatos(X[:,1], y, theta, costs)
    return theta

def calculaCosto(X, y, theta):
    """
    Calcula la función de costo para el modelo de regresión lineal

    J(theta) = 1/(2*m) * sum((h(x) - y)^2)

    Argumentos:
        X (vector): El vector de variables de entrada 
        y (vector): El vector de variables de salida reales
        theta (vector): El vector de parámetros calculados

    Regreso:
        float: El valor de la función de costo
    """
    return np.sum((X.dot(theta) - y) ** 2) / (2 * len(X))

def predice(X, theta):
    """
    Realiza una predicción con el modelo de regresión lineal

    Argumentos:
        X (vector): El vector de variables de entrada 
        theta (vector): El vector de parámetros calculados

    Regreso:
        float: El valor de la variable de salida
    """
    return X.dot(theta)

# Leer un archivo de texto con 2 variables, x e y
df = pd.read_csv('Proyecto1/examendata.txt', names=['X', 'y'])

# Crear una matriz de una columna de unos y una de la variable x
# Ayuda de Codeium para crear esta matriz
X = np.c_[np.ones(df.shape[0]), df['X']]
y = df['y']

theta = gradienteDescendiente(X, y, [0, 0], 0.01)

# Gradiente Descendiente para cuadraticas

#print(predict(np.array([1, 3.5]), theta))