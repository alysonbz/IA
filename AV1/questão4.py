#Importe as bibliotecas necessárias.
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
db = pd.read_csv('db_ajustado.csv')
print("\n Dataset: DB ATUALIZADO")
print(db)

#Normalize com a melhor normalização o conjunto de dados se houver melhoria.
X = db[['Glucose', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
y = db["Outcome"].values
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=11, stratify=y)

#Plote o gráfico com o a indicação do melhor k.
neighbors = np.arange(1, 10)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)

    knn.fit(X_train, y_train)

    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

print("acurácia no treino: ",train_accuracies, '\n',"Acuracia no teste: ", test_accuracies)
plt.title("Testando o melhor K")
plt.plot(neighbors, train_accuracies.values(), label="Treino acurácia")
plt.plot(neighbors, test_accuracies.values(), label="Teste acurácia")

plt.legend()
plt.xlabel("Número de vizinhos")
plt.ylabel("Acurácia")

plt.show()