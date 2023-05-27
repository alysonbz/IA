#bibliotecas
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
df_final= pd.read_csv("df_final.csv")
print(df_final)


# Separar os atributos de entrada (X) e o atributo alvo (y)
X = df_final.drop(['BodyFat'], axis=1)
y = df_final['BodyFat']
print(y)


X = df_final.drop(['BodyFat'], axis=1)
y = df_final['BodyFat'].values

#Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir os parâmetros para cada regressor
lasso_params = {'alpha': np.linspace(0.0001, 1, 20)}
ridge_params = {'alpha': np.linspace(0.01, 10, 20)}
# Realizar a codificação de atributos categóricos, se necessário
#label_encoder = LabelEncoder()
#X['Engine volume'] = label_encoder.fit_transform(X['Engine volume'])

# Inicializar os regressores
lasso = Lasso()
ridge = Ridge()
# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o objeto KFold para validação cruzada
kf = KFold(n_splits=6, shuffle=True, random_state=42)
# Definir os parâmetros para busca em grade
param_grid = {'alpha': [0.1, 1.0, 10.0]}

# Realizar busca em grade com validação cruzada para o regressor Lasso
lasso = Lasso()
lasso_cv = GridSearchCV(lasso, param_grid, cv=5)
lasso_cv.fit(X_train, y_train)

# Inicializar o objeto GridSearchCV para cada regressor
lasso_grid = GridSearchCV(lasso, lasso_params, cv=kf)
ridge_grid = GridSearchCV(ridge, ridge_params, cv=kf)
# Imprimir os melhores parâmetros e score para o regressor Lasso
print("Melhores parâmetros para Lasso:", lasso_cv.best_params_)
print("Melhor score para Lasso:", lasso_cv.best_score_)

# Realizar busca em grade com validação cruzada para o regressor Ridge
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid, cv=5)
ridge_cv.fit(X_train, y_train)

# Ajustar os modelos aos dados de treinamento
lasso_grid.fit(X_train, y_train)
ridge_grid.fit(X_train, y_train)
# Imprimir os melhores parâmetros e score para o regressor Ridge
print("Melhores parâmetros para Ridge:", ridge_cv.best_params_)
print("Melhor score para Ridge:", ridge_cv.best_score_)

# Imprimir as melhores configurações e scores para cada regressor
print("Melhores configurações para Lasso: ")
print(lasso_grid.best_params_)
print("Melhor score para Lasso: ")
print(lasso_grid.best_score_)

print("Melhores configurações para Ridge: ")
print(ridge_grid.best_params_)
print("Melhor score para Ridge: ")
print(ridge_grid.best_score_)