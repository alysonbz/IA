#Importe as bibliotecas necessárias.
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
df1_relevantes= pd.read_csv("df1_final.csv")
print(df1_relevantes)



#printNormalize o conjunto de dados com normalização logarítmica e verifique a acurácia do knn.
numeric_cols = df1_relevantes.select_dtypes(include="number").columns.drop("RainTomorrow")

df1_relevantes[numeric_cols] = df1_relevantes[numeric_cols].applymap(lambda x: np.log(x) if x > 0 else x)

df = df1_relevantes.replace([np.nan, np.inf, -np.inf], np.nan).dropna()
X = df1_relevantes.drop('RainTomorrow', axis=1)
y = df1_relevantes['RainTomorrow']


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
acuracia = knn.score(X_test, y_test)


#Normalize o conjunto de dados com normalização de media zero e variância unitária e e verifique a acurácia do knn.

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
print(X_norm.var())
print('variancia', X.var())

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)

# Criando o objeto KNN com k=5
knn = KNeighborsClassifier()

# Treinando o modelo
knn.fit(X_train, y_train)

# Verificando a acurácia nos dados de teste
accuracy = knn.score(X_test, y_test)

#Print as duas acuracias lado a lado para comparar.
print(acuracia)

print(accuracy)