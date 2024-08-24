import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px

def entrenaRN(input_layer_size, hidden_layer_size, num_labels, X, y, epochs=10000, learning_rate=0.02, momentum=0.9):
    """
    Función para la RN

    Argumentos:
        input_layer_size: Tamaño de la capa de entrada
        hidden_layer_size: Tamaño de la capa oculta
        num_labels: Tamaño de la capa de salida
        X: Matriz de datos de entrada
        y: Vector de datos de salida
        epochs: Cantidad de iteraciones
        learning_rate: Tasa de aprendizaje
        momentum: Momentum

    Regreso:
        dict: Diccionario con los pesos y bias de la RN
    """
    # Crear los pesos y bias para cada capa
    params = {'Wh': randinicializacionPesos(input_layer_size, hidden_layer_size),
              'bh': randinicializacionPesos(1, hidden_layer_size),
              'Wf': randinicializacionPesos(hidden_layer_size, num_labels),
              'bf': randinicializacionPesos(1, num_labels) }
        
    # Inicializar los cambios
    changes = {'Wh': np.zeros(params['Wh'].shape),
               'bh': np.zeros(params['bh'].shape),
               'Wf': np.zeros(params['Wf'].shape),
               'bf': np.zeros(params['bf'].shape)}
    
    # Constantes de paciencia
    best_J = np.inf
    paciencia = 10
    contador_paciencia = 0

    # Constantes cross-validation
    best_val_cross = 0
    pac_cross = 1000
    contador_cross = 0

    # Separar en datos de entrenamiento y validación
    porcentaje_val = 0.2
    cantidad_val = int(len(X) * porcentaje_val)
    X_val = X[0:cantidad_val, :]
    y_val = y[0:cantidad_val, :]

    X = X[cantidad_val:, :]
    y = y[cantidad_val:, :]

    # Constantes de regularización
    reg_lambda_l1 = 0.5
    reg_lambda_l2 = 0.5

    J = []
    val_cross = []
    for epoch in range(epochs):
        # Calcular la salida actual de la RN
        net_hidden = np.dot(X, params['Wh']) + params['bh']
        salida_hidden = sigmoidal(net_hidden)

        net_final = np.dot(salida_hidden, params['Wf']) + params['bf']
        salida_final = sigmoidal(net_final)

        # Calcular el coste
        J.append(np.mean(np.sum(-y * np.log(salida_final) - (1 - y) * np.log(1 - salida_final), axis=1)))

        # Calcular el error de la RN
        delta_salida = sigmoidalGradiente(net_final) * (y - salida_final)
        delta_hidden = sigmoidalGradiente(net_hidden) * (np.dot(delta_salida, params['Wf'].T))

        # Calcular los cambios
        changes = {'Wh': momentum * changes['Wh'] + learning_rate * np.dot(X.T, delta_hidden) / X.shape[0], # + reg_lambda_l2 * params['Wh'], # + reg_lambda_l1 * np.sign(params['Wh']),
                   'bh': momentum * changes['bh'] + learning_rate * np.sum(delta_hidden, axis=0, keepdims=True) / X.shape[0],
                   'Wf': momentum * changes['Wf'] + learning_rate * np.dot(salida_hidden.T, delta_salida) / X.shape[0], # + reg_lambda_l2 * params['Wf'], # + reg_lambda_l1 * np.sign(params['Wf']),
                   'bf': momentum * changes['bf'] + learning_rate * np.sum(delta_salida, axis=0, keepdims=True) / X.shape[0]}

        # Actualizar los pesos y bias con momento
        params['Wf'] += changes['Wf']
        params['bf'] += changes['bf']
        params['Wh'] += changes['Wh']
        params['bh'] += changes['bh'] 

        # Aplicar max-norm a los pesos
        max_norm = 3
        params['Wf'] = np.clip(params['Wf'], -max_norm, max_norm)
        params['Wh'] = np.clip(params['Wh'], -max_norm, max_norm)

        # Parar si el coste no mejora en las ultimas 10 iteraciones
        if J[-1] < best_J:
            best_J = J[-1]
            contador_paciencia = 0
        else:
            contador_paciencia += 1
        if contador_paciencia >= paciencia:
            print("Se ha parado debido a que el coste no ha mejorado en las ultimas", paciencia, "iteraciones")
            break

        # Cross-validation
        val_cross.append(calcular_precisiones(y_val, prediceRNYaEntrenada(X_val, params['Wh'], params['bh'], params['Wf'], params['bf'])))

        # Visualizar los datos
        #fig.add_scatter(x= np.array(range(epoch)), y= val_cross, name = "Validación")
        #scatter.update(x= np.array(range(epoch)), y = val_cross)


        # Parar si el valor de la validación no mejora en las ultimas 100 iteraciones
        """ if val_cross[-1] > best_val_cross:
            best_val_cross = val_cross[-1]
            contador_cross = 0
        else:
            contador_cross += 1
        if contador_cross >= pac_cross:
            print("Se ha parado debido a que el valor de la validación no ha mejorado en las ultimas", pac_cross, "iteraciones")
            break """

        if epoch % 100 == 0:
            print("Epoch:", epoch, "Cost:", J[-1] , "Porcentaje de precisión:", val_cross[-1] * 100, "%")

    print("Costo final:", J[-1])
    print("Iteraciones:", epoch)

    return params

def sigmoidalGradiente(z):
    """
    Calcula el gradiente de la sigmoidal de z

    sigmoidalGradiente = sigmoidal(z) * (1 - sigmoidal(z))

    Argumentos:
        z (float): El valor de z
    """
    return sigmoidal(z) * (1 - sigmoidal(z))

def sigmoidal(z):
    """
    Calcula la sigmoidal de z

    sigmoidal = 1 / (1 + e^(-z))

    Argumentos:
        z (float): El valor de z
    """
    # Evitar overflows
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def randinicializacionPesos(L_in, L_out, epsilon_init=0.12):
    """
    Crear pesos aleatorios utilizando la inicialización de Xavier

    Argumentos:
        L_in: Tamaño de la capa de entrada
        L_out: Tamaño de la capa de salida
        epsilon_init: Rango del valor aleatorio inicial

    Regreso:
        Array: Matriz de pesos del tamaño especificado por L_in y L_out
    """
    limite = np.sqrt(6 / (L_in + L_out))
    return np.random.uniform(low=-limite, high=limite, size=(L_in, L_out))

def calcular_precisiones(y, y_pred):
    """
    Calcula la precisión y la exhaustividad de una predicción

    Argumentos:
        y (list): Lista de valores reales
        y_pred (list): Lista de valores predichos

    Regreso:
        float: La precisión
        float: La exhaustividad
    """

    correct = 0
    for i in range(len(y)):
        if np.argmax(y_pred[i]) == np.argmax(y[i]):
            correct += 1

    precision = correct / len(y)

    return precision

def prediceRNYaEntrenada(X, Wh, bh, Wf, bf):
    """
    Realiza una predicción con la RN ya entrenada

    Argumentos:
        X: Matriz de datos de entrada
        Wh: Matriz de pesos de la primera capa
        bh: Vector de bias de la primera capa
        Wf: Matriz de pesos de la segunda capa
        bf: Vector de bias de la segunda capa

    Regreso:
        list: El valor calculado de cada ejemplo de X
    """
    net_hidden = np.dot(X, Wh) + bh
    salida_hidden = sigmoidal(net_hidden)
    net_final = np.dot(salida_hidden, Wf) + bf
    salida_final = sigmoidal(net_final)
    return salida_final

def main():
    # Obtener los datos
    X = np.loadtxt('Proyecto3/digitos.txt', delimiter=' ')
    
    # Randomizar el orden de los datos
    np.random.shuffle(X)

    # Convertir las salidas a one-hot encoding
    # https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-one-hot-encoded-array-in-numpy
    y = np.eye(10)[X[:, -1].astype(int)]
    X = X[:, 0:-1]

    # Calcular el tamaño de los datos de entrenamiento (80%)
    tamano_entrenamiento = int(len(X) * 0.8)

    # Separar en datos de entrenamiento y prueba
    X_train = X[0:tamano_entrenamiento, :]
    X_test = X[tamano_entrenamiento:, :]

    y_train = y[0:tamano_entrenamiento, :]
    y_test = y[tamano_entrenamiento:, :]

    #plt.imshow(X[0].reshape(20, 20), cmap='Greys')

    # Entrenar la RN
    w = entrenaRN(X_train.shape[1], 25, 10, X_train, y_train)

    # Obtener la precisión del modelo
    print(f'Precisión de datos de entrenamiento: {calcular_precisiones(y_train, prediceRNYaEntrenada(X_train, w['Wh'], w['bh'], w['Wf'], w['bf'])) * 100}%')
    print(f'Precisión de datos de prueba: {calcular_precisiones(y_test, prediceRNYaEntrenada(X_test, w['Wh'], w['bh'], w['Wf'], w['bf'])) * 100}%')

main()