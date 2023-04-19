#Importe as bibliotecas necessárias.
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np


#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
bt_atualizado = pd.read_csv('bt_novo.csv')
print("\n Dataset: Bt Atualizado")
print(bt_atualizado)

#Normalize o conjunto de dados com normalização logarítmica e verifique a acurácia do knn.

# Selecionar as colunas numéricas
colunas_numericas = bt_atualizado.select_dtypes(include='number').columns.drop('classe')
# Aplicar a logarítmica
bt_atualizado[colunas_numericas] = bt_atualizado[colunas_numericas].applymap(lambda x: np.log(x) if x > 0 else x)
# vendo se tem valores infinitos ou nulos na base de dados
print(bt_atualizado.isin([np.nan, np.inf, -np.inf]).sum())
# Tratando valores infinitos ou nulos
bt = bt_atualizado.replace([np.inf, -np.inf], np.nan).dropna()
print(bt, "\n")
X = bt.drop(['classe'], axis=1)
y = bt['classe'].values
knn = KNeighborsClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, stratify=y, random_state=5)
knn.fit(X_train,y_train)
acuracia_normalizada_log = knn.score(X_test,y_test)


#Normalize o conjunto de dados com normalização de media zero e variância unitária e e verifique a acurácia do knn.
X = bt_atualizado.drop(['classe'], axis=1)
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
y = bt_atualizado["classe"].values
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, stratify=y, random_state=5)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
acuracia_normalizada_0 = knn.score(X_train,y_train)


#Print as duas acuracias lado a lado para comparar.
print('Acurácia Normalizada com log e com scaler:')
print(acuracia_normalizada_log,acuracia_normalizada_0)