#Importe as bibliotecas necessárias.
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
df = pd.read_csv(r'C:\Users\Guilherme\Documents\G\IA\AV1\dataset\dataset__binary.csv')

#Normalize o conjunto de dados com normalização logarítmica  e verifique a acurácia do knn.
numeric_cols = df.select_dtypes(include='number').columns.drop('target') # seleciona as colunas numéricas

df[numeric_cols] = df[numeric_cols].applymap(lambda x: np.log(x) if x > 0 else x) # aplica a normalização logarítmica

#print(df.isin([np.nan, np.inf, -np.inf]).sum()) # verifica se existem valores infinitos ou nulos na base de dados

df = df.replace([np.inf, -np.inf], np.nan).dropna() # trata os valores infinitos ou nulos

X = df[numeric_cols].values # dataset com exceção da coluna "target" já normalizado
y = df['target'].values # obtem as labels da coluna "target"

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
knn = KNeighborsClassifier() # inicializa o algoritmo KNN
knn.fit(X_train, y_train) # aplica a função fit do KNN
#print("score scaler: ", knn.score(X_test, y_test))

#Normalize o conjunto de dados com normalização de media zero e variância unitária e e verifique a acurácia do knn.
df2 = pd.read_csv(r'C:\Users\Guilherme\Documents\G\IA\AV1\dataset\dataset__binary.csv')
scaler = StandardScaler() # inicializa o scale

X2 = df2.drop(['target'],axis=1) # exclue do dataset a coluna "target"
X2_norm = pd.DataFrame(scaler.fit_transform(X2), # normaliza o dataset com scaler
                      columns= X2.columns)
y2 = df2['target'].values # obtem as labels da coluna "target"

X2_train, X2_test, y2_train, y2_test = train_test_split(X2_norm, y2, stratify=y2, random_state=42) # divide o dataset em treino e teste com amostragem estratificada

knn.fit(X_train,y_train) # aplica a função fit do KNN
#print("score scaler: ", knn.score(X2_test, y2_test))

#Print as duas acuracias lado a lado para comparar.
print("score log: ",knn.score(X_test, y_test), "score scaler: ", knn.score(X2_test, y2_test))
