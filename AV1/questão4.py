#Importe as bibliotecas necessárias.
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
gender_one = pd.read_csv('gender_final.csv')
print("\n Dataset: Gender Atualizado")
print(gender_one)

#Normalize com a melhor normalização o conjunto de dados se houver melhoria.
X = gender_one
scaler = StandardScaler()
scaler.fit(X)
X_normalizado = scaler.fit_transform(X)
y = gender_one["gender"].values
X_train, X_test, y_train, y_test = train_test_split(X_normalizado, y, test_size=0.2, random_state=16)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Acurácia Normalizada:')
print(knn.score(X_test, y_test))

#Plote o gráfico com o a indicação do melhor k.
neighbors = np.arange(1, 26)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)

    knn.fit(X_train, y_train)

    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

print("acuracia no treino: ",train_accuracies, '\n',"Acuracia no teste: ", test_accuracies)
plt.title("Testando o melhor K")
plt.plot(neighbors, train_accuracies.values(), label="Traino acuracia")
plt.plot(neighbors, test_accuracies.values(), label="Teste acuracia")

plt.legend()
plt.xlabel("Numero de vizinhos")
plt.ylabel("Acuracia")

plt.show()

