import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from scipy.spatial import distance

# 1) Importe as bibliotecas necessárias
from sklearn.preprocessing import StandardScaler

# 2) Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
dataset_path = 'dataset/gender_classification_v7.csv'  # Atualize com o caminho correto
data = pd.read_csv(dataset_path)

# Supondo que as colunas 'feature1', 'feature2', ..., 'featureN' são as características e 'gender' é o rótulo
X = data.drop('gender', axis=1).values
y = data['gender'].values

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

# Normalização Logarítmica
X_log_normalized = np.log1p(X)
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log_normalized, y, test_size=0.3, random_state=42)

# Distância Euclidiana com normalização logarítmica
y_pred_log_euclidean = predict_classification(X_train_log, y_train_log, X_test_log, k, 'euclidean')
accuracy_log_euclidean = calculate_accuracy(y_test_log, y_pred_log_euclidean)
print(f'Acurácia com normalização logarítmica e distância Euclidiana: {accuracy_log_euclidean:.2f}')

# Normalização de Média Zero e Variância Unitária
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
X_train_standard, X_test_standard, y_train_standard, y_test_standard = train_test_split(X_standardized, y, test_size=0.3, random_state=42)

# Distância Euclidiana com normalização de média zero e variância unitária
y_pred_standard_euclidean = predict_classification(X_train_standard, y_train_standard, X_test_standard, k, 'euclidean')
accuracy_standard_euclidean = calculate_accuracy(y_test_standard, y_pred_standard_euclidean)
print(f'Acurácia com normalização de média zero e variância unitária e distância Euclidiana: {accuracy_standard_euclidean:.2f}')

# Print as duas acuracias lado a lado para comparar
print(f'Acurácia com normalização logarítmica e distância Euclidiana: {accuracy_log_euclidean:.2f}')
print(f'Acurácia com normalização de média zero e variância unitária e distância Euclidiana: {accuracy_standard_euclidean:.2f}')
