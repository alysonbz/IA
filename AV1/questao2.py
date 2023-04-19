#Importe as bibliotecas necessárias.

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.

airline = pd.read_csv('airline_ajustado.csv')
print(airline)

#Sem normalizar o conjunto de dados divida o dataset em treino e teste.

X = airline
#y = airline['satisfaction'].values_count() SOCORRO

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

#Implemente o Knn exbindo sua acurácia nos dados de teste e mantenha sua parametrização default.

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))