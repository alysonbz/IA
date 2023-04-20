#Importe as bibliotecas necessárias.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize, StandardScaler

#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
from questao1 import hda

#Normalize com a melhor normalização o conjunto de dados se houver melhoria.
X = hda.drop('HeartDisease', axis=1)
y = hda['HeartDisease']

knn=KNeighborsClassifier()
scaler = StandardScaler()

X_norm_scaler = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_norm_scaler, y, test_size=0.3, random_state=42, stratify=y)

knn.fit(X_train, y_train)
y_predscaler = knn.predict(X_test)
acc_scaler = accuracy_score(y_test, y_predscaler)

print('Acurácia com normalização de média zero e variância unitária: {:.2f}%'.format(acc_scaler * 100))

#Plote o gráfico com o a indicação do melhor k.
neighbors = np.arange(1, 12)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)

    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

print("\nAcurácia dos dados de treino: ",train_accuracies, '\n', '\nAcurácia dos dados de teste: ',test_accuracies)

plt.title("Número variável de vizinhos:")

plt.plot(neighbors, train_accuracies.values(), label="Acurácia do treino")
plt.plot(neighbors, test_accuracies.values(), label="Acurácia do teste")

plt.legend()
plt.xlabel("Numeros de vizinhos")
plt.ylabel("Acurácia")

plt.show()

