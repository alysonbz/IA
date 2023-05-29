# Importe as bibliotecas necessárias.
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
df = pd.read_csv(r'C:\Users\Guilherme\Documents\G\IA\AV1\dataset\dataset__binary.csv')

# Normalize com a melhor normalização o conjunto de dados se houver melhoria.
X = df.drop(['target'], axis=1)
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # divide o dataset em treino e teste

neighbors = np.arange(1, 9) # cria os vizinhos/neighbors
train_accuracies = {} # cria uma lista para as acuracias de treino
test_accuracies = {} # cria uma lista para as acuracias de treino

for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor) # inicializa o algoritmo KNN

    knn.fit(X_train, y_train) # ajusta o modelo

    train_accuracies[neighbor] = knn.score(X_train, y_train) # computa as acuracias
    test_accuracies[neighbor] = knn.score(X_test, y_test) # computa as acuracias

print("acurácias de treino: ",train_accuracies, '\n',"acurácias de teste: ", test_accuracies) # printa as acuracias

plt.title("KNN: Variando o Número de Vizinhos") # adiciona o titulo
plt.plot(neighbors, train_accuracies.values(), label="Acurácia de Treino") # plota acuracias de treino
plt.plot(neighbors, test_accuracies.values(), label="Acurácia de Teste") # palot acuracias de teste
plt.legend()
plt.xlabel("Número de Vizinhos") # plota a legenda do eixo x
plt.ylabel("Acurácia") # plota a legenda do eixo y
plt.show() # exibe o grafico