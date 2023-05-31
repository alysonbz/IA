import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Carregar o dataset atualizado
data = pd.read_csv("forest_fires_updated.csv")
scaler = StandardScaler()

# Separar os atributos e o alvo
relevant_columns = ['month', 'day', 'ISI', 'temp', 'wind']
X = data[relevant_columns].values
y = data['area'].values

X_norm = scaler.fit_transform(X)


# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

# Definir os parâmetros para busca em grade
param_grid_lasso = {'alpha': [0.1, 1.0, 10.0]}
param_grid_ridge = {'alpha': [0.1, 1.0, 10.0], 'solver': ['lsqr', 'sag']}


# Realizar busca em grade com validação cruzada para o regressor Lasso
lasso = Lasso()
lasso_cv = GridSearchCV(lasso, param_grid_lasso, cv=5)
lasso_cv.fit(X_train, y_train)

# Imprimir os melhores parâmetros e score para o regressor Lasso
print("Melhores parâmetros para Lasso:", lasso_cv.best_params_)
print("Melhor score para Lasso:", lasso_cv.best_score_)

# Realizar busca em grade com validação cruzada para o regressor Ridge
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid_ridge, cv=5)
ridge_cv.fit(X_train, y_train)

# Imprimir os melhores parâmetros e score para o regressor Ridge
print("Melhores parâmetros para Ridge:", ridge_cv.best_params_)
print("Melhor score para Ridge:", ridge_cv.best_score_)