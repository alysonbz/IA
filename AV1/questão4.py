# Importe as bibliotecas necessárias.
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado
df = pd.read_csv('Hotel_Reservations_ajustado.csv')

# Normalize com a melhor normalização o conjunto de dados se houver melhoria.

X = df.drop('booking_status', axis=1)

y = df['booking_status'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Plote o gráfico com o a indicação do melhor k.

neighbors = np.arange(1, 12)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:

    knn = KNeighborsClassifier(n_neighbors=neighbor)

    knn.fit(X_train, y_train)

    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)


print("accuracy: ",train_accuracies, '\n',"acuracy de teste: ", test_accuracies)


plt.title("KNN: Numero variavel de vizinhos")

plt.plot(neighbors, train_accuracies.values(), label="Treino Accuracy")

plt.plot(neighbors, test_accuracies.values(), label="Teste Accuracy")

plt.legend()
plt.xlabel("Numero de vizinhos")
plt.ylabel("Accuracy")

plt.show()

