# Importe as bibliotecas necessárias.
import pandas as pd
from scipy.spatial.distance import mahalanobis, chebyshev, cityblock
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter

# Carregue o dataset definido para você.
healthcare_data = pd.read_csv('dataset/healthcare-dataset-stroke-data-new.csv')
print(healthcare_data.head())

# Sem normalizar o conjunto de dados divida o dataset em treino e teste.
X = healthcare_data.drop(columns=['stroke'])
y = healthcare_data['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Função para calcular a distância euclidiana
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

def chebyshev_distance(a, b):
    return np.max(np.abs(a - b))

def mahalanobis_distance(a, b, VI):
    return mahalanobis(a, b, VI)

# Função KNN manual
def knn_predict(X_train, y_train, X_test, k, distance_func, VI=None):
    predictions = []
    for test_point in X_test:
        distances = []
        for i, train_point in enumerate(X_train):
            if distance_func == mahalanobis_distance:
                distance = distance_func(test_point, train_point, VI)
            else:
                distance = distance_func(test_point, train_point)
            distances.append((distance, y_train[i]))
        # Ordenar as distâncias e selecionar os k vizinhos mais próximos
        distances.sort(key=lambda x: x[0])
        k_nearest_neighbors = distances[:k]
        # Determinar a classe majoritária entre os k vizinhos mais próximos
        classes = [neighbor[1] for neighbor in k_nearest_neighbors]
        majority_class = Counter(classes).most_common(1)[0][0]
        predictions.append(majority_class)
    return predictions

# Preparando os dados para a predição
X_test_np = X_test.to_numpy()
X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()

# Calculando a matriz inversa da covariância para a distância de Mahalanobis
VI = np.linalg.inv(np.cov(X_train_np.T))

# Calculando a acurácia para cada métrica de distância
distances = {
    'Euclidiana': euclidean_distance,
    'Manhattan': manhattan_distance,
    'Chebyshev': chebyshev_distance,
    'Mahalanobis': mahalanobis_distance
}

accuracies = {}

for name, func in distances.items():
    if name == 'Mahalanobis':
        y_pred = knn_predict(X_train_np, y_train_np, X_test_np, k=1, distance_func=func, VI=VI)
    else:
        y_pred = knn_predict(X_train_np, y_train_np, X_test_np, k=1, distance_func=func)
    accuracy = np.mean(y_pred == y_test.to_numpy())
    accuracies[name] = accuracy

# Calculando a acurácia manualmente
accuracy = np.mean(y_pred == y_test.to_numpy())

# Exibindo a acurácia
print(f'Acurácia do KNN manual: {accuracy}')
print()

# Exibindo as acurácias
for name, accuracy in accuracies.items():
    print(f'Acurácia do KNN com distância {name}: {accuracy}')
