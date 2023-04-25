#Importe as bibliotecas necessárias.
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
db = pd.read_csv("db_ajustado.csv")


#Normalize o conjunto de dados com normalização logarítmica e verifique a acurácia do knn.
# Selecionar as colunas numéricas
numeric_cols = db.select_dtypes(include='number').columns.drop('Outcome')

# Aplicar a normalização logarítmica
db[numeric_cols] = db[numeric_cols].applymap(lambda x: np.log(x) if x > 0 else x)

# Verificar se existem valores infinitos ou nulos na base de dados
print(db.isin([np.nan, np.inf, -np.inf]).sum())

# Tratar os valores infinitos ou nulos
db = db.replace([np.inf, -np.inf], np.nan).dropna()
print(db, "\n")
X = db[numeric_cols].values
y = db['Outcome'].values
knn = KNeighborsClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size=0.2, random_state=11)
knn.fit(X_train,y_train)
acuracia_log = knn.score(X_test, y_test)


#Normalize o conjunto de dados com normalização de media zero e variância unitária e e verifique a acurácia do knn.

X = db[['Glucose', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
y = db["Outcome"].values

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, stratify=y,  test_size=0.2, random_state=11)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
acuracia_scaler = knn.score(X_test, y_test)


#Print as duas acuracias lado a lado para comparar.
print('Acurácia Normalizada com log e com scaler:')
print(acuracia_log, acuracia_scaler)
