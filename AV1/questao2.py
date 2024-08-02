# Importar as bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

# Carregar o dataset
file_path = 'dataset/flavors_of_cacao_ajustado.csv'
df = pd.read_csv(file_path)

# Separar o dataset em características e rótulo
X = df.drop(columns=['Rating'])
y = df['Rating']

# Converter a coluna de rótulos em valores numéricos
y = y.astype('category').cat.codes

# Converter todas as colunas de X para valores numéricos
X = X.apply(lambda col: col.astype('category').cat.codes if col.dtype == 'object' else col)

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Função para predição KNN
def knn_predict(X_train, y_train, X_test, k, distance_metric):
    y_pred = []
    for test_point in X_test.values:
        distances = []
        for i, train_point in enumerate(X_train.values):
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
                raise ValueError("Métrica de distância não suportada")
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

# Mostrar acurácia para cada métrica
for dist, acc in accuracies.items():
    print(f'Acurácia com distância {dist}: {acc:.2f}')

