#neste segundo exercício você deve realizar uma classificação utilizando KNN implementado de forma manual.

#Importe as bibliotecas necessárias.
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from src.utils import load_cancer_dataset_cleaned

#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
cancer = load_cancer_dataset_cleaned()

print("2.1) Sem normalizar o conjunto de dados divida o dataset em treino e teste.")
#primeiro é separado as características(x) e as classes Alvo(y)
x = cancer.drop(columns=['diagnosis'])
y = cancer['diagnosis']
#print("caracteristicas: \n",x)
#print("classe alvo: \n",y)

#treino e testes
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state= 42)
print("Foi separado a Classe Alvo(y) Diagnosis das caracteristicas(x). \nApós isso, o dataset é dividido em conjuntos de treino e teste. \nNo código acima, 20% dos dados são reservados para o teste, e 80% para o treino. ")


print("2.2) Implemente o Knn exbindo sua acurácia nos dados de teste")

# Função para calcular a distância Euclidiana
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Função para calcular a distância Manhattan
def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))

# Função para calcular a distância Chebyshev
def chebyshev_distance(point1, point2):
    return np.max(np.abs(point1 - point2))

# Função para calcular a distância Mahalanobis
def mahalanobis_distance(point1, point2, VI):
    delta = point1 - point2
    return np.sqrt(np.dot(np.dot(delta, VI), delta))

# Função para encontrar os k vizinhos mais próximos com uma métrica de distância específica
def get_k_nearest_neighbors(X_train, y_train, test_point, k=3, distance_metric='euclidean', VI=None):
    distances = []
    for i in range(len(X_train)):
        if distance_metric == 'euclidean':
            distance_value = euclidean_distance(X_train.iloc[i], test_point)
        elif distance_metric == 'manhattan':
            distance_value = manhattan_distance(X_train.iloc[i], test_point)
        elif distance_metric == 'chebyshev':
            distance_value = chebyshev_distance(X_train.iloc[i], test_point)
        elif distance_metric == 'mahalanobis':
            distance_value = mahalanobis_distance(X_train.iloc[i], test_point, VI)
        distances.append((distance_value, y_train.iloc[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    return neighbors

# Função para prever a classe de uma amostra
def predict(X_train, y_train, test_point, k=3, distance_metric='euclidean', VI=None):
    neighbors = get_k_nearest_neighbors(X_train, y_train, test_point, k, distance_metric, VI)
    classes = [neighbor[1] for neighbor in neighbors]
    majority_vote = Counter(classes).most_common(1)[0][0]
    return majority_vote

# Função para calcular acurácia com diferentes métricas
def calculate_accuracy(X_train, X_test, y_train, y_test, k=3, distance_metric='euclidean', VI=None):
    predictions = [predict(X_train, y_train, X_test.iloc[i], k, distance_metric, VI) for i in range(len(X_test))]
    accuracy = np.sum(predictions == y_test.values) / len(y_test)
    return accuracy

# Cálculo da matriz inversa para Mahalanobis
VI = np.linalg.inv(np.cov(x_train.T))

# Calcular a acurácia para cada métrica de distância
accuracy_euclidean = calculate_accuracy(x_train, x_test, y_train, y_test, k=3, distance_metric='euclidean')
accuracy_manhattan = calculate_accuracy(x_train, x_test, y_train, y_test, k=3, distance_metric='manhattan')
accuracy_chebyshev = calculate_accuracy(x_train, x_test, y_train, y_test, k=3, distance_metric='chebyshev')
accuracy_mahalanobis = calculate_accuracy(x_train, x_test, y_train, y_test, k=3, distance_metric='mahalanobis', VI=VI)

print("2.3) Compare as acurácias considerando 4 possíveis cálculos de distancias diferentes: \n a) distância de mahalanobis. b) distancia de chebyshev c) distância de manhattan d) distancia euclidiana")
print(f"Acurácia usando Distância Euclidiana: {accuracy_euclidean * 100:.2f}%")
print(f"Acurácia usando Distância Manhattan: {accuracy_manhattan * 100:.2f}%")
print(f"Acurácia usando Distância Chebyshev: {accuracy_chebyshev * 100:.2f}%")
print(f"Acurácia usando Distância Mahalanobis: {accuracy_mahalanobis * 100:.2f}%")

# No geral, parece que o KNN com distâncias Euclidiana, Manhattan, Chebyshev
# funciona bem no dataset, enquanto a distância Mahalanobis pode não ser ideal para esta aplicação específica.
