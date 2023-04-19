#Importe as bibliotecas necessárias.
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
bt_atualizado = pd.read_csv('bt_novo.csv')
print("\n Dataset: Bt Atualizado")
print(bt_atualizado)

#Normalize com a melhor normalização o conjunto de dados se houver melhoria.
X = bt_atualizado.drop(['classe'], axis=1)
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
y = bt_atualizado["classe"].values
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=5)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
acuracia_normalizada_0 = knn.score(X_train,y_train)

#Plote o gráfico com o a indicação do melhor k.
neighbors = np.arange(1, 16)
train_accuracies = {}
test_accuracies = {}
for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)
print("Acurácia no treino: ",train_accuracies, '\n',"Acuracia no teste: ", test_accuracies)
plt.title("Testando o melhor K")
plt.plot(neighbors, train_accuracies.values(), label="Treino acurácia")
plt.plot(neighbors, test_accuracies.values(), label="Teste acurácia")
plt.legend()
plt.xlabel("Numero de vizinhos")
plt.ylabel("Acuracia")
plt.show()
print('\n Percebemos que o melhor K está entre 1-4 e 8-10 ')