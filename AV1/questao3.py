#Importe as bibliotecas necessárias.
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
cancer = pd.read_csv('dados_preprocessados.csv')
print(cancer)
X = pd.read_csv('dados_preprocessados2.csv')
y = pd.read_csv('dados_preprocessados3.csv')


#Normalize o conjunto de dados com normalização logarítmica e verifique a acurácia do knn.
X_norm = np.log(X)
print(X_norm)

knn = KNeighborsClassifier(6)

print('score', knn.score(X_test, y_test))
#Normalize o conjunto de dados com normalização de media zero e variância unitária e e verifique a acurácia do knn.


#Print as duas acuracias lado a lado para comparar.