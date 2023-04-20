
#Importe as bibliotecas necessárias.
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
flavors_of_cacao = pd.read_csv('flavors_of_cacao_ajustado.csv')


#Transforma a variável "Rating" em uma variável categórica
rating_bins = [0, 2.5, 3.5, 4.0, 5.0]
rating_labels = ['ruim', 'regular', 'bom', 'excelente']
flavors_of_cacao['Rating_cat'] = pd.cut(flavors_of_cacao['Rating'], bins=rating_bins, labels=rating_labels)


#Normalize com a melhor normalização o conjunto de dados se houver melhoria.

X = flavors_of_cacao.drop(['Rating', 'Rating_cat'], axis=1)
y = flavors_of_cacao['Rating_cat']

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
accuracia_var = knn.score(X_test, y_test)


# Plote o gráfico com o a indicação do melhor k.
neighbors = np.arange(1, 15)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
    knn =KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)

    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

print("\nTreino, acuracia: ",train_accuracies)
print("\nteste, acuracia: ", test_accuracies)
plt.title("Qual o melhor k possível:")
plt.plot(neighbors, train_accuracies.values(), label="Acurácia de treino")
plt.plot(neighbors, test_accuracies.values(), label="Acurácia de teste")
plt.legend()
plt.show()

