import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_initial(X):
    # Crear una gráfica con dos subplots
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], marker='o')

def plot_final(X, centroids, labels, centroid_path=None):
    plt.subplot(1, 2, 2)

    # PLot the data
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=labels, cmap='prism')

    # Plot all the centroids in centroid_path if it is not None
    if centroid_path is not None:
        for j in range(centroid_path.shape[1]):
            plt.scatter(centroid_path[:, j, 0], centroid_path[:, j, 1], marker='x', c='gray', s=200)

        # Plot the track of the centroids as lines
        for j in range(centroid_path.shape[1]):
            plt.plot(centroid_path[:, j, 0], centroid_path[:, j, 1], 'k-')
                
    # Plot the final centroids as x's
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='black', s=200)

def find_closest_centroids(X, initial_centroids, return_distances=False):
    """
    Obtiene los índices de los centroides más cercanos a cada punto de los datos X

    Argumentos:
        X (vector): El vector de datos
        initial_centroids (vector): El vector de centroides iniciales
        return_distances (bool): Si se retornan las distancias calculadas, o solo los índices

    Regreso:
        vector: El vector de índices
    """
    distances = []
    #distances = np.array([[np.linalg.norm(x - centroid) for centroid in centroids] for x in X])
    
    for x in X:
        for centroid in initial_centroids:
            distances.append(np.linalg.norm(x - centroid))

    # Convertir la lista de distancias en un arreglo de tamaño (n, k)
    distances = np.array(distances).reshape(len(X), len(initial_centroids))
    
    # Asignar cada punto a su centroide correspondiente, el mas cercano
    labels = np.argmin(distances, axis=1)

    if return_distances:
        return [labels, distances]
    return labels


def compute_centroids(X, idx, k):
    """
    Calcula los centros de los grupos de datos

    Argumentos:
        X (vector): El vector de datos
        idx (vector): El vector de índices
        k (int): El número de grupos
    
    Regreso:
        vector: El vector de centros
    """
    new_centroids = []
     # Actualizar los centroides
    for i in range(k):
        # Calcula la media de todos los puntos asignados a este centroide
        new_centroids.append(X[idx == i].mean(axis=0))

    # Convertir la lista de centroides en un arreglo
    new_centroids = np.array(new_centroids)

    return new_centroids

def runkMeans(X, initial_centroids, max_iters, verbose=False, return_labels=False):
    """
    Ejecuta el algoritmo de k-means
    
    Argumentos:
        X (vector): El vector de datos
        initial_centroids (vector): El vector de centroides iniciales
        max_iters (int): El número máximo de iteraciones
        verbose (bool): Si se imprime el progreso de los centroides para cada iteración
        return_labels (bool): Si se retornan los índices de los centroides
    
    Regreso:
        vector: El vector de centroides finales
    """
    if verbose:
        plot_initial(X)
    best_error = float('inf')
    for _ in range(max_iters):

        if verbose:
            centroid_path = []

        iters = 0
        while True:
            [idx, distances] = find_closest_centroids(X, initial_centroids, return_distances=True)
            new_c = compute_centroids(X, idx, initial_centroids.shape[0])

            # Guardar los centroides para cada iteración
            if verbose:
                centroid_path.append(initial_centroids)

            # Verificar si los centroides no cambian, si es asi, se rompe el ciclo
            if np.all(initial_centroids == new_c):
                print('Converged after {} iterations'.format(iters))
                break

            iters += 1
            initial_centroids = new_c

        # Calcular el error
        error = np.sum(np.min(distances, axis=1))

        # Guardar el mejor resultado
        if error < best_error:
            best_error = error
            best_labels = idx
            best_centroids = new_c
            if verbose:
                best_centroid_path = np.array(centroid_path)

    if verbose:
        plot_final(X, best_centroids, best_labels, best_centroid_path)
        plt.show()

    if return_labels:
        return [best_centroids, best_labels]
    return best_centroids

def kMeansInitCentroids(X, K):
    """
    Obtenemos K centroides iniciales aleatoriamente a partir de los datos X

    Argumentos:
        X (vector): El vector de datos
        K (int): El número de centroides iniciales
    
    Regreso:
        vector: El vector de centroides iniciales
    """
    return X[np.random.choice(X.shape[0], K, replace=False)]

def main():
    # Leer el archivo de texto
    X = np.loadtxt('Proyecto4/ex7data2.txt', delimiter=' ')

    # Ejecutar k-means
    #runkMeans(X, kMeansInitCentroids(X, 3), 10, verbose=True)

    # Ejecutar k-means con centroides iniciales predefinidos
    init_centroids = np.array(([3, 3], [6, 2], [8, 5]))
    runkMeans(X, init_centroids, 10, verbose=True)

def image_compress():
    # Leer una imagen en formato PNG
    image = Image.open('Proyecto4/bird_small.png')
    image_array = np.array(image)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_array)

    # Convertir la imagen a 3 vectores
    flat_image = image_array.reshape(image_array.shape[0] * image_array.shape[1], 3)

    # Ejecutar k-means para cada pixel, obteniendo 16 grupos (colores)
    [colors, pixels] = runkMeans(flat_image, kMeansInitCentroids(flat_image, 16), 1, return_labels=True)

    # Reconstruir la imagen de los resultados del k-means
    new_image = np.array([colors[pixel] for pixel in pixels])

    # Reconvertir la imagen al tamaño original
    new_image_array = np.array(new_image).astype(int).reshape(image_array.shape[0], image_array.shape[1], 3)

    plt.subplot(1, 2, 2)
    plt.imshow(new_image_array)
    plt.show()

image_compress()