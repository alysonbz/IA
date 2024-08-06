import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler

ClanWaterQuality = pd.read_csv('dataset/CleanWaterQuality1.csv')

scaler = StandardScaler()
X = ClanWaterQuality[["aluminium","ammonia","arsenic","barium","cadmium","chloramine","chromium","copper","flouride","bacteria",
                      "viruses","lead","nitrates","nitrites","mercury","perchlorate","radium","selenium","silver","uranium"]].values
y = ClanWaterQuality["is_safe"].values



# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Aplique a transformação aos dados de treino e teste
transformer = FunctionTransformer(np.log1p)

# Aplique a transformação aos dados de treino e teste
X_train_norm = transformer.fit_transform(X_train)
X_test_norm = transformer.fit_transform(X_test)

dt_train = pd.DataFrame(X_train)
dt_test = pd.DataFrame(X_test)

X_train_scaled = pd.DataFrame(scaler.fit_transform(dt_train), columns=dt_train.columns)
X_test_scaled = pd.DataFrame(scaler.fit_transform(dt_test), columns = dt_test.columns)
# Create neighbors
neighbors = np.arange(1, 12)
train_accuracies = {}
test_accuracies = {}

train_accuracies_norm = {}
test_accuracies_norm = {}

train_accuracies_scaled = {}
test_accuracies_scaled = {}


for neighbor in neighbors:
    # Set up a KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=neighbor)

    # Fit the model
    knn.fit(X_train, y_train)
    # Compute accuracy
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

    train_accuracies_norm[neighbor] = knn.score(X_train_norm, y_train)

for neighbor in neighbors:

    knn = KNeighborsClassifier(n_neighbors=neighbor)

    knn.fit(X_train_norm, y_train)

    train_accuracies_norm[neighbor] = knn.score(X_train_norm, y_train)
    test_accuracies_norm[neighbor] = knn.score(X_test_norm, y_test)


for neighbor in neighbors:

    knn = KNeighborsClassifier(n_neighbors=neighbor)

    knn.fit(X_train_scaled, y_train)

    train_accuracies_scaled[neighbor] = knn.score(X_train_scaled, y_train)
    test_accuracies_scaled[neighbor] = knn.score(X_test_scaled, y_test)


print("acuracy on train: ",train_accuracies, '\n',"acuracy on test: ", test_accuracies)

print("acuracy on train norm: ",train_accuracies_norm, '\n',"acuracy on test norm: ", test_accuracies_norm)

print("acuracy on train scaled: ",train_accuracies_scaled, '\n',"acuracy on test scaled: ", test_accuracies_scaled)

#graficos

plt.figure(figsize = (10,6))
plt.title("KNN: K-Nearest Neighbors")

plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")

plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")

plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

plt.show()

#logaritmica

plt.figure(figsize = (10,6))
plt.title("KNN: K-Nearest Neighbors")

plt.plot(neighbors, train_accuracies_norm.values(), label="Training Accuracy")

plt.plot(neighbors, test_accuracies_norm.values(), label="Testing Accuracy")

plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy Normalized")

plt.show()

#scaler

plt.figure(figsize = (10,6))
plt.title("KNN: K-Nearest Neighbors")

plt.plot(neighbors, train_accuracies_scaled.values(), label="Training Accuracy")

plt.plot(neighbors, test_accuracies_scaled.values(), label="Testing Accuracy")

plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy Scaled")

plt.show()