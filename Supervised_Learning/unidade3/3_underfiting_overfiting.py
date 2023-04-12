from src.utils import load_churn_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

churn_df = load_churn_dataset()
X = churn_df[["account_length",  "total_day_charge" , "total_eve_charge",  "total_night_charge","total_intl_charge","number_customer_service_calls"]].values
y = churn_df["churn"].values

# Dividindo em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Criando vizinhos
neighbors = np.arange(1, 12)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
    # Configurando um classificador KNN
    knn = KNeighborsClassifier(n_neighbors=neighbor)

    # Ajuste do modelo
    knn.fit(X_train, y_train)

    # Precisão de cálculo
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)


print("\nAcurácia (precisão) do treino: ",train_accuracies, '\n',"Acurácia (precisão) do teste: ", test_accuracies)

# Adicionando o título
plt.title("KNN: Variação do número dos vizinhos")

# Plotando a precisão do treinamento
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")

# Plotando a precisão de teste
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")

plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

# Exibir o gráfico
plt.show() # melhor k = 7