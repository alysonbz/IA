#Importe as bibliotecas necessárias.
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
cancer = pd.read_csv('dados_preprocessados.csv')
#print(cancer)



#printNormalize o conjunto de dados com normalização logarítmica e verifique a acurácia do knn.
numeric_cols = cancer.select_dtypes(include="number").columns.drop("diagnosis")

cancer[numeric_cols] = cancer[numeric_cols].applymap(lambda x: np.log(x) if x > 0 else x)

X = cancer.drop('diagnosis', axis=1)
y = cancer['diagnosis']


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
acuracia = knn.score(X_test, y_test)


#Normalize o conjunto de dados com normalização de media zero e variância unitária e e verifique a acurácia do knn.

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
print(X_norm.var())   #variancia de X_normalizado
print('variancia', X.var())  #variancia de X

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)

# Criando o objeto KNN com k=5
knn = KNeighborsClassifier()

# Treinando o modelo
knn.fit(X_train, y_train)

# Verificando a acurácia nos dados de teste
accuracy_norm = knn.score(X_test, y_test)

#Print as duas acuracias lado a lado para comparar.
print(acuracia)


print(accuracy_norm)