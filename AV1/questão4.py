#Importe as bibliotecas necessárias.
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
df1_relevantes = pd.read_csv("df1_final.csv")

#Normalize com a melhor normalização o conjunto de dados se houver melhoria.

X = df1_relevantes.drop('RainTomorrow', axis=1)
y = df1_relevantes['RainTomorrow']

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)
print("não houve melhoria")
# Criando o objeto KNN com k=5
knn = KNeighborsClassifier(5)

# Treinando o modelo
knn.fit(X_train, y_train)

# Verificando a acurácia nos dados de teste
accuracy_norm = knn.score(X_test, y_test)
print(accuracy_norm)

# Plote o gráfico com o a indicação do melhor k.
neighbors = np.arange(1, 10)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)

    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

print("\nAcurácia de treino: ",train_accuracies)
print("\nAcurácia de teste: ", test_accuracies)
plt.title("Vendo qual o melhor k:")
plt.plot(neighbors, train_accuracies.values(), label="Acurácia de treino")
plt.plot(neighbors, test_accuracies.values(), label="Acurácia de teste")
plt.legend()
plt.show()
print("melhor k é o 7")

