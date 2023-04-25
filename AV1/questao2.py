import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
df1_relevantes = pd.read_csv("df1_final.csv")
print(df1_relevantes)

#Sem normalizar o conjunto de dados divida o dataset em treino e teste.

X = df1_relevantes.drop('RainTomorrow', axis=1)
y = df1_relevantes['RainTomorrow']

# Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train, X_test, y_train, y_test)

#Implemente o Knn exbindo sua acurácia nos dados de teste e mantenha sua parametrização default.
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_test, y_test)

print('score', knn.score(X_test, y_test))


