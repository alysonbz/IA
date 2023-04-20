import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from questao2 import X_train, X_test, y_train, y_test, X, y,  knn

# Carregar o dataset
df = pd.read_csv('Drugs_and_Features')  # substitua 'dataset.csv' pelo nome do seu arquivo de dataset

# Dividir o dataset em features (X) e target (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Dividir o dataset em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar a normalização aos dados (opcional)
scaler = StandardScaler()  # substitua pelo tipo de normalização desejada (ex: MinMaxScaler())
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Parametrização do KNN
k_values = list(range(1, 11))  # valores de K a serem testados
train_accuracies = []
test_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    train_accuracy = knn.score(X_train_scaled, y_train)
    test_accuracy = knn.score(X_test_scaled, y_test)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Plotar o gráfico com os resultados
plt.plot(k_values, train_accuracies, label="Train Accuracy")
plt.plot(k_values, test_accuracies, label="Test Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.title("Acurácia do KNN em função do valor de K")
plt.grid(True)
plt.show()

# Identificar o melhor valor de K
best_k = k_values[test_accuracies.index(max(test_accuracies))]
print(f'O melhor valor de K é: {best_k}')
