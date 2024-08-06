#Neste exercicio vc deve buscar saber a melhor parametrização do knn.

#Importe as bibliotecas necessárias.
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from src.utils import load_cancer_dataset_cleaned
#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
cancer = load_cancer_dataset_cleaned()
#4.1) Normalize com a melhor normalização o conjunto de dados se houver melhoria.
x = cancer.drop(columns=['diagnosis'])
y = cancer['diagnosis']

#Aplicando a normalização Logarítimica
x_log_normalized = np.log1p(x)

#treino e testes
x_train, x_test, y_train, y_test = train_test_split(x_log_normalized, y, test_size=0.2, random_state=42)

k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

#4.2) Plote o gráfico com o a indicação do melhor k.
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
plt.title('Acurácia do KNN para diferentes valores de k')
plt.xlabel('Valor de k')
plt.ylabel('Acurácia')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Imprimir o melhor valor de k
best_k = k_values[np.argmax(accuracies)]
best_accuracy = max(accuracies)
print(f"O melhor valor de k é {best_k} com uma acurácia de {best_accuracy * 100:.2f}%")