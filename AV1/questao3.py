import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from questao2 import X_train, X_test, y_train, y_test, X, y,  knn


#carregar o Dataset

Drug_and_Features = pd.read_csv('Drugs_and_Features')
print(Drug_and_Features)


# Inicializer o scale
scaler = StandardScaler()

X = Drug_and_Features.drop(['Drug'],axis=1)

#normalize o dataset com scaler
X_norm = pd.DataFrame(scaler.fit_transform(X),
                      columns= X.columns)

#obtenha as labels da coluna Drugs
y = Drug_and_Features['Drug']

#print a variância de X
print('variancia: \n', np.var(X))

#print a variânca do dataset X_norm
print('variancia do dataset normalizado: \n', np.var(X_norm))

# Divida o dataset em treino e teste com amostragem estratificada
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, stratify=y, random_state=42)

#inicialize o algoritmo KNN
knn = KNeighborsClassifier()

# Aplique a função fit do KNN
knn.fit(X_train,y_train)

# Verifique o acerto do classificador
t=knn.score(X_test, y_test)


#Normalização Logaritmica

# Verificar se há valores infinitos em cada coluna do DataFrame X
for col in X.columns:
    if np.isinf(X[col]).any():
        # Tratar os valores infinitos, por exemplo, substituindo por NaN
        X[col].replace([np.inf, -np.inf], np.nan, inplace=True)

# Verificar se há valores muito grandes em cada coluna do DataFrame X
for col in X.columns:
    if np.max(np.abs(X[col])) > np.finfo('float64').max:
        # Tratar os valores muito grandes, por exemplo, substituindo por NaN
        X[col].replace(np.max(np.abs(X[col])), np.nan, inplace=True)


X = Drug_and_Features.drop(['Drug'], axis=1)

y = Drug_and_Features['Drug'].values

# divida o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

knn = KNeighborsClassifier()

# Aplique a função fit do knn
knn.fit(X_train, y_train)

# mostre o acerto do algoritmo
t_log = knn.score(X_test, y_test)

print("Score com normalização de media zero e variância unitária", t, "and ", "dados com normalização logarítmica", t_log )
print("Nenhuma das duas normalizações conseguio obter uma acurácia tão superior aos dados não normalizados...")