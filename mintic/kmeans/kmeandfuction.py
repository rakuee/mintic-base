from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = fetch_ucirepo(id=53)

# Datos
X = dataset.data.features
y = dataset.data.targets


print(dataset.metadata)


print(dataset.variables)

print(X.head())
print(y.head())

def euclidian_distance(point1,point2):
    # ---- 1) Validación de tipo iterable ------------------------------
    try:
        p1 = list(point1)
        p2 = list(point2)
    except TypeError:
        raise TypeError("Ambos argumentos deben ser iterables (list, tuple, np.ndarray o fila de DataFrame).")

    # ---- 2) Validación de dimensiones --------------------------------
    if len(p1) != len(p2):
        raise ValueError(
            f"Los puntos deben tener la misma dimensión. "
            f"Recibido: len(point1)={len(p1)}, len(point2)={len(p2)}."
        )

    if len(p1) == 0:
        raise ValueError("Los puntos no pueden estar vacíos.")

    # ---- 3) Validación de valores numéricos --------------------------
    try:
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
    except (ValueError, TypeError):
        raise ValueError("Todos los valores de los puntos deben ser numéricos.")

    # ---- 4) Cálculo de la distancia euclidiana -----------------------
    # Implementación manual: raíz cuadrada de la suma de cuadrados
    suma_cuadrados = 0.0
    for a, b in zip(p1, p2):
        suma_cuadrados += (a - b) ** 2

    distancia = suma_cuadrados ** 0.5
    return float(distancia)

def create_kmeans(data, n_clusters=8, max_iter=1000, tol=0.0001,randomstate=None):


    # ---------- 1) Convertir data a numpy array ----------
    if isinstance(data, pd.DataFrame):
        X = data.values
    else:
        X = np.asarray(data)

    # ---------- 2) Validaciones ----------
    if X.ndim != 2: 
        raise ValueError("data debe ser una matriz 2D (n_samples, n_features).")
    n_samples, n_features = X.shape

    if not isinstance(n_clusters, int) or n_clusters <= 0:
        raise ValueError("n_clusters debe ser un entero positivo.")
    if n_clusters > n_samples:
        raise ValueError("n_clusters no puede ser mayor que el número de muestras.")
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter debe ser un entero positivo.")
    if tol < 0:
        raise ValueError("tol debe ser no negativo.")

    # ---------- 3) Inicialización aleatoria de centroides ----------
    rng = np.random.default_rng(randomstate)          # Generador moderno de numpy
    indices = rng.choice(n_samples, size=n_clusters, replace=False)
    centroids = X[indices].copy()                     # Copia para no modificar los datos originales

    # ---------- 4) Representación del modelo ----------
    model = {
        'centroids': centroids,
        'n_clusters': n_clusters,
        'max_iter': max_iter,
        'tol': tol,
        'randomstate': randomstate,
        'data': X,
        'n_features': n_features,
        'n_samples': n_samples
    }

    return model

def kmeans_fit(kmeans, verbose=True):
    # ---------- 1) Validar que la representación tenga las claves requeridas ----------
    required_keys = {'centroids', 'data', 'n_clusters', 'max_iter', 'tol', 'n_features'}
    missing = required_keys - set(kmeans.keys())
    if missing:
        raise KeyError(f"La representación de K-Means no contiene las claves: {missing}")

    X = np.asarray(kmeans['data'], dtype=float)
    centroids = np.asarray(kmeans['centroids'], dtype=float).copy()
    n_clusters = kmeans['n_clusters']
    max_iter = kmeans['max_iter']
    tol = kmeans['tol']
    n_samples, n_features = X.shape

    if centroids.shape != (n_clusters, n_features):
        raise ValueError(
            f"Los centroides tienen forma {centroids.shape} y se esperaba "
            f"({n_clusters}, {n_features})."
        )

    labels = np.zeros(n_samples, dtype=int)
    converged = False

    # ---------- 2) Bucle principal ----------
    for iteration in range(1, max_iter + 1):
        # -------- 2.1) Asignación: cada punto al centroide más cercano --------
        # Distancia euclidiana vectorizada (equivalente a euclidean_distance punto a punto)
        # ||x - c||^2 = ||x||^2 - 2 x·c + ||c||^2
        X_norm = np.sum(X ** 2, axis=1, keepdims=True)          # (n_samples, 1)
        C_norm = np.sum(centroids ** 2, axis=1, keepdims=True)  # (n_clusters, 1)
        cross = X @ centroids.T                                 # (n_samples, n_clusters)
        dist_sq = X_norm - 2 * cross + C_norm.T                 # (n_samples, n_clusters)
        dist_sq = np.clip(dist_sq, a_min=0.0, a_max=None)       # evitar negativos numéricos
        labels = np.argmin(dist_sq, axis=1)

        # -------- 2.2) Actualización: recalcular centroides --------
        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            mask = (labels == k)
            if np.any(mask):
                new_centroids[k] = X[mask].mean(axis=0)
            else:
                # Cluster vacío: re-inicializar con un punto aleatorio para no perderlo
                new_centroids[k] = X[np.random.randint(n_samples)]

        # -------- 2.3) Calcular desplazamiento total de centroides --------
        shift = np.sum(np.sqrt(np.sum((new_centroids - centroids) ** 2, axis=1)))

        if verbose:
            print(f"Iteración {iteration:4d} | desplazamiento total de centroides = {shift:.6f}")

        centroids = new_centroids

        # -------- 2.4) Criterio de convergencia --------
        if shift < tol:
            converged = True
            if verbose:
                print(f"✔ Convergencia alcanzada en la iteración {iteration} "
                      f"(desplazamiento {shift:.6f} < tol {tol}).")
            break

    # ---------- 3) Cálculo de la inercia final ----------
    X_norm = np.sum(X ** 2, axis=1, keepdims=True)
    C_norm = np.sum(centroids ** 2, axis=1, keepdims=True)
    cross = X @ centroids.T
    dist_sq = np.clip(X_norm - 2 * cross + C_norm.T, a_min=0.0, a_max=None)
    inertia = float(np.sum(dist_sq[np.arange(n_samples), labels]))

    if not converged and verbose:
        print(f"⚠ Se alcanzó el máximo de iteraciones ({max_iter}) sin converger.")

    # ---------- 4) Guardar resultados en la representación ----------
    kmeans['centroids'] = centroids
    kmeans['labels'] = labels
    kmeans['inertia'] = inertia
    kmeans['n_iter'] = iteration
    kmeans['converged'] = converged

    return kmeans

def inertia(kmeans, X):
    X = np.asarray(X)
    centros = np.asarray(kmeans['centroids'], dtype=float)[kmeans['labels']]  # centroide de cada punto
    diferencias = X - centros
    return float(np.sum(diferencias ** 2))

def elbow(data, min=2, max=8, max_iter=1000, tol=0.001, randomstate=None):
    # ---------- 1) Validación de parámetros ----------
    if not isinstance(min, int) or not isinstance(max, int):
        raise TypeError("'min' y 'max' deben ser enteros.")
    if min < 1:
        raise ValueError("'min' debe ser al menos 1.")
    if max < min:
        raise ValueError("'max' debe ser mayor o igual que 'min'.")

    # ---------- 2) Entrenar un K-Means por cada k ----------
    ks = list(range(min, max + 1))
    inercias = []

    for k in ks:
        # Creamos la representación del modelo
        km = create_kmeans(
            data,
            n_clusters=k,
            max_iter=max_iter,
            tol=tol,
            randomstate=randomstate,
        )
        # Entrenamos (sin imprimir el log de cada iteración)
        km = kmeans_fit(km, verbose=False)
        # Calculamos la inercia final
        inercias.append(inertia(km, km['data']))

    # ---------- 3) Graficar ----------
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inercias, marker="o", linewidth=2)
    plt.title("Método del Codo")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inercia")
    plt.xticks(ks)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return None