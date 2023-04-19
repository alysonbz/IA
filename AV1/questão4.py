# Importe as bibliotecas necessárias.
from questao1 import avc_ajustado
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
avc_ajustado = avc_ajustado

# Normalize com a melhor normalização o conjunto de dados se houver melhoria.
X = avc_ajustado.drop('stroke', axis=1)
y = avc_ajustado['stroke']
scaler = StandardScaler()
knn = KNeighborsClassifier()
X_normalizado_scaler = scaler.fit_transform(X)
X_train_scaler, X_test_scaler, y_train_scaler, y_test_scaler = train_test_split(X_normalizado_scaler, y, test_size=0.3, random_state=1, stratify=y)

knn.fit(X_train_scaler, y_train_scaler)
y_pred_scaler = knn.predict(X_test_scaler)
acuracia_scaler = accuracy_score(y_test_scaler, y_pred_scaler)
print('Acurácia com a melhor normalização: a scaler.\nResultado: {:.2f}%'.format(acuracia_scaler * 100))

# Plote o gráfico com o a indicação do melhor k.
neighbors = np.arange(2, 18)
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
plt.plot(neighbors, train_accuracies.values(), label="Acurácia de treino")
plt.plot(neighbors, test_accuracies.values(), label="Acurácia de teste")
plt.legend()
plt.show()
print("\nMelhor K: igual ou maior a 8")

