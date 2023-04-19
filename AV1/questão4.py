# Importe as bibliotecas necessárias.
from questao1 import customer_ajustado
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize, StandardScaler

# Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
customer_ajustado = customer_ajustado


# Normalize com a melhor normalização o conjunto de dados se houver melhoria.
X = customer_ajustado.drop('label', axis=1)
y = customer_ajustado['label']
scaler = StandardScaler()
knn=KNeighborsClassifier()
X_norm_scaler = scaler.fit_transform(X)
X_train_scaler, X_test_scaler, y_train_scaler, y_test_scaler = train_test_split(X_norm_scaler, y, test_size=0.3, random_state=42, stratify=y)

knn.fit(X_train_scaler, y_train_scaler)
y_pred_scaler = knn.predict(X_test_scaler)
acc_scaler = accuracy_score(y_test_scaler, y_pred_scaler)

print('Acurácia com normalização de média zero e variância unitária: {:.2f}%'.format(acc_scaler * 100))



# Plote o gráfico com o a indicação do melhor k.
neighbors = np.arange(1, 12)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train_scaler, y_train_scaler)

    train_accuracies[neighbor] = knn.score(X_train_scaler, y_train_scaler)
    test_accuracies[neighbor] = knn.score(X_test_scaler, y_test_scaler)

print("\nAcurácia dos dados de treino: ",train_accuracies)
print("\nAcurácia dos dados de teste: ", test_accuracies)
plt.title("Testando o melhor K:")
plt.plot(neighbors, train_accuracies.values(), label="Acurácia do treino")
plt.plot(neighbors, test_accuracies.values(), label="Acurácia do teste")
plt.legend()
plt.show()
print("O melhor K é o 4")

