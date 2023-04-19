#Importe as bibliotecas necessárias.
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
cancer = pd.read_csv('dados_preprocessados.csv')
print(cancer)
X = pd.read_csv('dados_preprocessados2.csv')
y = pd.read_csv('dados_preprocessados3.csv')

#Normalize o conjunto de dados com normalização logarítmica e verifique a acurácia do knn.
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)



#Normalize o conjunto de dados com normalização de media zero e variância unitária e e verifique a acurácia do knn.


#Print as duas acuracias lado a lado para comparar.