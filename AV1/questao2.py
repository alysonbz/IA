import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from scipy.spatial import distance

# 2) Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
dataset_path = 'dataset/gender_classification_v7.csv'  # Atualize com o caminho correto
data = pd.read_csv(dataset_path)

# Supondo que as colunas 'feature1', 'feature2', ..., 'featureN' são as características e 'gender' é o rótulo
X = data.drop('gender', axis=1).values
y = data['gender'].values

# 3) Sem normalizar o conjunto de dados, divida o dataset em treino e teste.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Função para calcular a matriz de covariância inversa, necessária para a distância de Mahalanobis
def inverse_covariance_matrix(X):
    cov_matrix = np.cov(X, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    return inv_cov_matrix

# Função para calcular a distância de Mahalanobis manualmente
def mahalanobis_distance(u, v, inv_cov_matrix):
    delta = u - v
    m_dist = np.sqrt(np.dot(np.dot(delta, inv_cov_matrix), delta.T))
    return m_dist

# Função para calcular a distância entre dois pontos com base no tipo de distância especificado
def calculate_distance(u, v, distance_type, inv_cov_matrix=None):
    if distance_type == 'euclidean':
        return np.linalg.norm(u - v)
    elif distance_type == 'chebyshev':
        return np.max(np.abs(u - v))
    elif distance_type == 'manhattan':
        return np.sum(np.abs(u - v))
    elif distance_type == 'mahalanobis' and inv_cov_matrix is not None:
        return mahalanobis_distance(u, v, inv_cov_matrix)
    else:
        raise ValueError("Tipo de distância não suportado ou parâmetros inválidos.")

# Função para encontrar os k vizinhos mais próximos
def get_k_nearest_neighbors(X_train, y_train, test_sample, k, distance_type, inv_cov_matrix=None):
    distances = []
    for i in range(len(X_train)):
        dist = calculate_distance(test_sample, X_train[i], distance_type, inv_cov_matrix)
        distances.append((dist, y_train[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    return neighbors

# Função para prever a classe com base nos k vizinhos mais próximos
def predict_classification(X_train, y_train, X_test, k, distance_type, inv_cov_matrix=None):
    predictions = []
    for test_sample in X_test:
        neighbors = get_k_nearest_neighbors(X_train, y_train, test_sample, k, distance_type, inv_cov_matrix)
        output_values = [neighbor[1] for neighbor in neighbors]
        prediction = Counter(output_values).most_common(1)[0][0]
        predictions.append(prediction)
    return predictions

# Função para calcular a acurácia
def calculate_accuracy(y_test, y_pred):
    correct = np.sum(y_test == y_pred)
    accuracy = correct / len(y_test)
    return accuracy

# Número de vizinhos
k = 5

# 4) e 5) Implementar e comparar as acurácias com diferentes distâncias
# a) Distância de Mahalanobis
inv_cov_matrix = inverse_covariance_matrix(X_train)
y_pred_mahalanobis = predict_classification(X_train, y_train, X_test, k, 'mahalanobis', inv_cov_matrix)
accuracy_mahalanobis = calculate_accuracy(y_test, y_pred_mahalanobis)
print(f'Acurácia com distância de Mahalanobis: {accuracy_mahalanobis:.2f}')

# b) Distância de Chebyshev
y_pred_chebyshev = predict_classification(X_train, y_train, X_test, k, 'chebyshev')
accuracy_chebyshev = calculate_accuracy(y_test, y_pred_chebyshev)
print(f'Acurácia com distância de Chebyshev: {accuracy_chebyshev:.2f}')

# c) Distância de Manhattan
y_pred_manhattan = predict_classification(X_train, y_train, X_test, k, 'manhattan')
accuracy_manhattan = calculate_accuracy(y_test, y_pred_manhattan)
print(f'Acurácia com distância de Manhattan: {accuracy_manhattan:.2f}')

# d) Distância Euclidiana
y_pred_euclidean = predict_classification(X_train, y_train, X_test, k, 'euclidean')
accuracy_euclidean = calculate_accuracy(y_test, y_pred_euclidean)
print(f'Acurácia com distância Euclidiana: {accuracy_euclidean:.2f}')
