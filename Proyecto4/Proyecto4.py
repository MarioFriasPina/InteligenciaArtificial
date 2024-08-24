import numpy as np
import matplotlib.pyplot as plt

def plot_initial(X):

    # Crear una gráfica con dos subplots
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], marker='o')

def plot_final(X, centroids, labels):
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=labels, cmap='prism')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='black', s=200)

def k_means(X, k, max_iterations=10):
    plot_initial(X)

    best_error = float('inf')
    for _ in range(max_iterations):

        # Iniciar los k centroides aleatoriamente a partir de los datos
        centroids = X[np.random.choice(X.shape[0], k, replace=False)]

        # Iterar hasta converger
        while True:
            # Calcular la distancia de cada punto a cada centroide utilizando la norma de la distancia Euclidiana

            distances = []
            #distances = np.array([[np.linalg.norm(x - centroid) for centroid in centroids] for x in X])
            
            for x in X:
                for centroid in centroids:
                    distances.append(np.linalg.norm(x - centroid))

            # Convertir la lista de distancias en un arreglo de tamaño (n, k)
            distances = np.array(distances).reshape(len(X), len(centroids))
            
            # Asignar cada punto a su centroide correspondiente, el mas cercano
            labels = np.argmin(distances, axis=1)

            new_centroids = []
            #new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
            
            # Actualizar los centroides
            for i in range(k):
                # Calcula la media de todos los puntos asignados a este centroide
                new_centroids.append(X[labels == i].mean(axis=0))

            # Convertir la lista de centroides en un arreglo
            new_centroids = np.array(new_centroids)

            # Verificar convergencia
            if np.array_equal(centroids, new_centroids):
                break
            
            #Actualiza los centroides con la nueva media calculada
            centroids = new_centroids
        
        # Calcular el error
        error = np.sum(np.min(distances, axis=1))

        # Guardar el mejor resultado
        if error < best_error:
            print(f'Error: {error}')
            best_error = error
            best_labels = labels
            best_centroids = centroids
            
    plot_final(X, best_centroids, best_labels)
    plt.show()

    return [centroids, error]

# Read the dataset
X = np.loadtxt('Proyecto4/ex7data2.txt', delimiter=' ')

k_means(X, 3)