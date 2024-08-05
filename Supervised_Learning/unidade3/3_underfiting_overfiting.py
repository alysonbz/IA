from src.utils import load_churn_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

churn_df = load_churn_dataset()
X = churn_df[["account_length",  "total_day_charge" , "total_eve_charge",  "total_night_charge","total_intl_charge","number_customer_service_calls"]].values
y = churn_df["churn"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create neighbors
neighbors = np.arange(1, 21) # **1 a 20** vizinhos
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
    # Set up a KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=neighbor) # **n_neighbors** recebe o número de vizinhos

    # Fit the model
    knn.fit(X_train, y_train) # **X_train** e **y_train** para o treinamento

    # Compute accuracy
    train_accuracies[neighbor] = knn.score(X_train, y_train) # **X_train** e **y_train** para a acurácia de treinamento
    test_accuracies[neighbor] = knn.score(X_test, y_test) # **X_test** e **y_test** para a acurácia de teste

print("acuracy on train: ",train_accuracies, '\n',"acuracy on test: ", test_accuracies)

# Add a title
plt.title("Acurácia do KNN com Variação do Número de Vizinhos")

# Plot training accuracies
plt.plot(neighbors, train_accuracies.values(), label="Acurácia de Treinamento")

# Plot test accuracies
plt.plot(neighbors, test_accuracies.values(), label="Acurácia de Teste")

plt.legend()
plt.xlabel("Número de Vizinhos")
plt.ylabel("Acurácia")

# Display the plot
plt.show()
