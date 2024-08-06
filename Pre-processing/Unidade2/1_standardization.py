from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

wine = load_wine_dataset()

print(wine)
print()

X = wine.drop(['Quality'], axis=1)

y = wine['Quality'].values

# divida o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

knn = KNeighborsClassifier()

# Aplique a função fit do knn para calcular distâncias
knn.fit(X_train, y_train)

pred = knn.predict(X_test)
# mostre o acerto do algoritmo
print(knn.score(X_test, y_test)) # Calcular distâncias de cada ponto de teste

print("knn result: ", pred, "\n")
print("label: ", y_test)