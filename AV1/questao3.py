#Importe as bibliotecas necessárias.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from  sklearn.preprocessing import StandardScaler
from src.utils import load_wine_dataset

scaler = StandardScaler()
knn = knn = KNeighborsRegressor()



#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.]
star_class_new = pd.read_csv(r"C:\Users\Aluno\Downloads\BIANCA\IA\AV1\dataset\star_classification.csv")
print(star_class_new)


#Normalize o conjunto de dados com normalização logarítmica e verifique a acurácia do knn.
X = star_class_new.drop(["class"], axis=1)
y = star_class_new[["class"]].values

X_norm = scaler.star_class_new(X)
print('variancia do dataset normalizado', X_norm.var())
print(X.var())
print(np.log(star_class_new["classe"]))#normalização logaritma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
knn.fit(X_train,y_train)
print("acuracia do Knn", knn.score(X_test, y_test))


#Normalize o conjunto de dados com normalização de media zero e variância unitária e e verifique a acurácia do knn.
X = star_class_new.drop(["class"], axis=1)
y = star_class_new[["class"]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# criando
neighbors = np.arange(1, 12)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:

    knn = KNeighborsRegressor(n_neighbors=neighbor)

    # Fit the model
    knn.fit(X_train, y_train)
    print(knn.score(X_test, y_test))

    # Compute accuracy
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

print(knn.score(X_test, y_test))
#Print as duas acuracias lado a lado para comparar.
print("acuracy on train: ",train_accuracies, '\n',"acuracy on test: ", test_accuracies)