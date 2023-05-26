#bibliotecas
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
dm = pd.read_csv('dados.csv')
print(dm)

# Separar os atributos de entrada (X) e o atributo alvo (y)
X = dm.drop(['Price'], axis=1)
y = dm['Price']


#Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir os parâmetros para cada regressor
lasso_params = {'alpha': np.linspace(0.0001, 1, 20)}
ridge_params = {'alpha': np.linspace(0.01, 10, 20)}

# Inicializar os regressores
lasso = Lasso()
ridge = Ridge()

# Inicializar o objeto KFold para validação cruzada
kf = KFold(n_splits=6, shuffle=True, random_state=42)


# Inicializar o objeto GridSearchCV para cada regressor
lasso_grid = GridSearchCV(lasso, lasso_params, cv=kf)
ridge_grid = GridSearchCV(ridge, ridge_params, cv=kf)

# Ajustar os modelos aos dados de treinamento
lasso_grid.fit(X_train, y_train)
ridge_grid.fit(X_train, y_train)

# Imprimir as melhores configurações e scores para cada regressor
print("Melhores configurações para Lasso: ")
print(lasso_grid.best_params_)
print("Melhor score para Lasso: ")
print(lasso_grid.best_score_)

print("Melhores configurações para Ridge: ")
print(ridge_grid.best_params_)
print("Melhor score para Ridge: ")
print(ridge_grid.best_score_)