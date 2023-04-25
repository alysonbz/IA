#Importe as bibliotecas necessárias.
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
gender_one = pd.read_csv('gender_final.csv')
print("\n Dataset: Gender Atualizado")
print(gender_one)

#Normalize o conjunto de dados com normalização logarítmica e verifique a acurácia do knn.

#Pegando somente as colunas numericas
colunasnumericas= gender_one.select_dtypes(include='number').columns.drop('gender')

#Normalizando com log
gender_one[colunasnumericas]= gender_one[colunasnumericas].applymap(lambda x: np.log(x) if x > 0 else x)

#Indentificando e Tratando valores nulos ou infinitos
print= (gender_one.isin([np.nan, np.inf, -np.inf]).sum())
gender_one= gender_one.replace([np.inf, -np.inf], np.nan).dropna()
print = (gender_one, "\n")
X=gender_one[colunasnumericas].values
y=gender_one['gender'].values
knn= KNeighborsClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
knn.fit(X_train, y_train)
acuracia_log = knn.score(X_test, y_test)

#Normalize o conjunto de dados com normalização de media zero e variância unitária e e verifique a acurácia do knn.
X = gender_one[['long_hair','forehead_height_cm','nose_wide', 'nose_long']].values
scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X)
y = gender_one['gender'].values
X_train, X_test, y_train, y_test = train_test_split(X_normalizado, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
acuracia_scaler = knn.score(X_test, y_test)

#Print as duas acuracias lado a lado para comparar.
print('Acurácia Normalizada:')
print(acuracia_log, acuracia_scaler)