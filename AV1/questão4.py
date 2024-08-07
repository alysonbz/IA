
# 1) Importe as bibliotecas necessárias. 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 2) Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
data = pd.read_csv('diabetes_atualizado.csv')
print(data.head())

# 3) Normalize com a melhor normalização o conjunto de dados se houver melhoria.
X = data.drop('Outcome', axis=1)
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4) Plote o gráfico com a indicação do melhor k.
k_values = range(1, 21)

accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

plt.plot(k_values, accuracies)
plt.xlabel('Valor de k')
plt.ylabel('Acurácia')
plt.title('Acurácia do KNN para Diferentes Valores de k')
plt.show()

best_k = k_values[accuracies.index(max(accuracies))]
print(f'O melhor valor de k é: **{best_k}**')