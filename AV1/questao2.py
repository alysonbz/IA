import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import mahalanobis, chebyshev, cityblock, euclidean
import numpy as np

# Carregue o dataset (substitua 'diabetes_atualizado.csv' pelo nome do seu arquivo)
diabetes_atualizado = pd.read_csv('diabetes_atualizado.csv')
print(diabetes_atualizado.head())

# Divida o dataset em treino e teste
X = diabetes_atualizado.drop('Outcome', axis=1)
y = diabetes_atualizado['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def knn_predict(X_train, y_train, X_test, k, distance_metric):
    y_pred = []
    for test_point in X_test.values:
        distances = []
        for i, train_point in enumerate(X_train.values):
            if distance_metric == 'mahalanobis':
                # Calcula a matriz de covariância dos dados de treinamento
                cov_matrix = np.cov(X_train.T)
                dist = mahalanobis(test_point, train_point, cov_matrix)
            elif distance_metric == 'chebyshev':
                dist = chebyshev(test_point, train_point)
            elif distance_metric == 'manhattan':
                dist = cityblock(test_point, train_point)
            elif distance_metric == 'euclidean':
                dist = euclidean(test_point, train_point)
            else:
                raise ValueError('Métricas de distância inválida.')
            distances.append((dist, y_train.iloc[i]))
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]
        # Calcula a classe mais frequente entre os k vizinhos mais próximos
        classes = [neighbor[1] for neighbor in k_nearest]
        predicted_class = max(set(classes), key=classes.count)
        y_pred.append(predicted_class)
    return y_pred

# Define o valor de k
k = 5

# Calcula a acurácia para cada métrica de distância
for distance_metric in ['mahalanobis', 'chebyshev', 'manhattan', 'euclidean']:
    y_pred = knn_predict(X_train, y_train, X_test, k, distance_metric)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia com {distance_metric} distance: {accuracy:.4f}')
#5) Compare as acurácias considerando que 4 possíveis cálculos de distancias diferentes:
   #a) distância de mahalanobis.

   #Acurácia com mahalanobis distance: 0.7208

   #b) distancia de chebyshev

   #Acurácia com chebyshev distance: 0.6753

   #c) distância de manhattan

   #Acurácia com manhattan distance:0.7403

   #d) distancia euclidiana

   #Acurácia com euclidean distance: 0.7208
