import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from src.utils import load_new_heart_dataset

# Carregue o dataset definido para você.
heart = load_new_heart_dataset()

# Separar features e target
X = heart.drop('output', axis=1)
y = heart['output']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def knn_predict(X_train, y_train, X_test, k, dist_metric):
    predictions = []
    for test_point in X_test.values:
        # Calcular distâncias
        distances = []
        for train_point, label in zip(X_train.values, y_train):
            if dist_metric == 'mahalanobis':
                d = distance.mahalanobis(test_point, train_point, np.linalg.inv(np.cov(X_train.T)))
            elif dist_metric == 'chebyshev':
                d = distance.chebyshev(test_point, train_point)
            elif dist_metric == 'manhattan':
                d = distance.cityblock(test_point, train_point)
            elif dist_metric == 'euclidean':
                d = distance.euclidean(test_point, train_point)
            distances.append((d, label))

        # Ordenar distâncias e obter os k vizinhos mais próximos
        distances.sort(key=lambda x: x[0])
        neighbors = [label for _, label in distances[:k]]

        # Predição
        prediction = max(set(neighbors), key=neighbors.count)
        predictions.append(prediction)

    return np.array(predictions)


# Configurar o valor de k
k = 5

# Predições usando diferentes métricas de distância
for metric in ['mahalanobis', 'chebyshev', 'manhattan', 'euclidean']:
    y_pred = knn_predict(X_train, y_train, X_test, k, metric)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia com distância {metric}: {accuracy:.4f}')