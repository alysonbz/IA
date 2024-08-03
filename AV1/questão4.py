import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Carregue o dataset
dataset_path = 'dataset/gender_classification_v7.csv'
data = pd.read_csv(dataset_path)

# Supondo que as colunas 'long_hair', 'forehead_width_cm', etc. são as características e 'gender' é o rótulo
X = data.drop('gender', axis=1).values
y = data['gender'].values

# Normalização com média zero e variância unitária
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.3, random_state=42)

# Busca pelo melhor valor de k
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    accuracies.append(accuracy)

# Plote o gráfico com a indicação do melhor k
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.title('Acurácia em função do número de vizinhos (k)')
plt.xlabel('Número de vizinhos (k)')
plt.ylabel('Acurácia')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Melhor valor de k
best_k = k_values[np.argmax(accuracies)]
print(f'O melhor valor de k é {best_k} com acurácia de {max(accuracies):.2f}')
