#Importe as bibliotecas necessárias.
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
cancer_relevancia = pd.read_csv('dados_preprocessados.csv')


#print Normalize o conjunto de dados com normalização logarítmica e verifique a acurácia do knn.
numeric_cols = cancer_relevancia.select_dtypes(include="number").columns.drop("diagnosis")

cancer_relevancia[numeric_cols] = cancer_relevancia[numeric_cols].applymap(lambda x: np.log(x) if x > 0 else x)

X = cancer_relevancia.drop('diagnosis', axis=1)#pega todas as colunas exceto diagnosis e atribui a X
y = cancer_relevancia['diagnosis']  #pega somente a coluna diagnosis e atribui a y


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
acuracia_norm_log = knn.score(X_test, y_test)


#Normalize o conjunto de dados com normalização de media zero e variância unitária e e verifique a acurácia do knn.

scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_normalizado, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
accuracia_var = knn.score(X_test, y_test)

#Print as duas acuracias lado a lado para comparar.
print(acuracia_norm_log)

print(accuracia_var)