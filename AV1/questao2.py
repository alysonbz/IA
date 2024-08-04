### Questão 2
"""
Neste segundo exercício você deve realizar uma classificação utilizando KNN implementado de forma manual.

#### Instruções
1) Importe as bibliotecas necessárias.
2) Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
3) Sem normalizar o conjunto de dados divida o dataset em treino e teste.
4) Implemente o Knn exbindo sua acurácia nos dados de teste
5) Compare as acurácias considerando que 4 possíveis cálculos de distancias diferentes:
   a) distância de mahalanobis.
   b) distancia de chebyshev
   c) distância de manhattan
   d) distancia euclidiana
"""
from scipy.spatial import distance

# Importe as bibliotecas necessárias.
from src.utils import load_new_customer_dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregue o dataset novo.
new_customer = load_new_customer_dataset()
print(new_customer.head())

# Divida o dataset em treino e teste, sem normalizar.
X = new_customer.drop(columns=['label'])
y = new_customer['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Implemente o KNN exibindo sua acurácia nos dados de teste.
def knn_predict(X_train, y_train, X_test, k, distance_metric):
    y_pred = []
    for test_point in X_test:
        distances = []
        for i, train_point in enumerate(X_train):
            if distance_metric == 'euclidean':
                dist = np.linalg.norm(test_point - train_point)
            elif distance_metric == 'manhattan':
                dist = np.sum(np.abs(test_point - train_point))
            elif distance_metric == 'chebyshev':
                dist = np.max(np.abs(test_point - train_point))
            elif distance_metric == 'mahalanobis':
                vi = np.linalg.inv(np.cov(X_train.T))
                dist = distance.mahalanobis(test_point, train_point, vi)
            distances.append((dist, y_train[i]))
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
    y_pred = knn_predict(X_train.values, y_train.values, X_test.values, k, dist)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[dist] = accuracy

for dist, acc in accuracies.items():
    print(f'Acurácia com distância {dist}: {acc:.2f}')

