# Importe as bibliotecas necessárias.
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

# Carregue o dataset definido para você.
drug200_new = pd.read_csv('dataset/drug200_new.csv')
print(drug200_new.head())

# Divida o dataset em treino e teste.
X = drug200_new.drop(columns=['Drug'])
y = drug200_new['Drug']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Implemente o KNN exibindo sua acurácia nos dados de teste
def knn_predict(X_train, y_train, X_test, k, distance_metric):
    y_pred = []
    for test_point in X_test.values:
        distances = []
        for i, train_point in enumerate(X_train.values):
            # Inicialize a distância
            dist = None

            if distance_metric == 'euclidean':
                dist = np.linalg.norm(test_point - train_point)
            elif distance_metric == 'manhattan':
                dist = np.sum(np.abs(test_point - train_point))
            elif distance_metric == 'chebyshev':
                dist = np.max(np.abs(test_point - train_point))
            elif distance_metric == 'mahalanobis':
                vi = np.linalg.inv(np.cov(X_train.T))
                dist = distance.mahalanobis(test_point, train_point, vi)
            else:
                raise ValueError(f"Unknown distance metric: {distance_metric}")

            distances.append((dist, y_train.iloc[i]))

        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]
        neighbor_labels = [label for _, label in neighbors]
        most_common = max(set(neighbor_labels), key=neighbor_labels.count)
        y_pred.append(most_common)

    return np.array(y_pred)


# Predições e cálculos de acurácia para diferentes métricas de distância
k = 5

distances = ['euclidean', 'manhattan', 'chebyshev', 'mahalanobis']
accuracies = {}

for dist in distances:
    y_pred = knn_predict(X_train, y_train, X_test, k, dist)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[dist] = accuracy

for dist, acc in accuracies.items():
    print(f'Acurácia com distância {dist}: {acc:.2f}')
