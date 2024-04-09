from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
#teste1

wine = load_wine_dataset()

X = wine.drop(['Quality'],axis=1)

y = wine['Quality'].values

# divida o dataset em treino e teste
X_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

knn = KNeighborsClassifier()

# Aplique a função fit do knn
knn.fit(x_test, y_test)

# mostre o acerto do algoritmo
print(knn.score(x_test, y_test))