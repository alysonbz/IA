import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import Lasso, Ridge

# Carregar o dataset
data = pd.read_csv(r'C:\Users\eryka\Downloads\archive\Samsung Electronics.csv')

# Selecionar os atributos relevantes
relevant_attributes = ['High', 'Low', 'Close']

# Filtrar o DataFrame com os atributos relevantes
data_relevant = data[relevant_attributes]

# Separar os dados em atributos e alvo
X = data[['Close']]
y = data[['Close']]

# Definir os valores a serem testados para os parâmetros de regularização
lasso_params = {'alpha': [0.1, 1.0, 10.0]}
ridge_params = {'alpha': [0.1, 1.0, 10.0]}

# Criar objetos dos regressores Lasso e Ridge
lasso = Lasso()
ridge = Ridge()

# Criar objeto de busca de grade com validação cruzada
lasso_grid = GridSearchCV(lasso, lasso_params, cv=5)
ridge_grid = GridSearchCV(ridge, ridge_params, cv=5)

# Ajustar o modelo de Lasso aos dados
lasso_grid.fit(X, y)

# Ajustar o modelo de Ridge aos dados
ridge_grid.fit(X, y)

# Imprimir as melhores configurações e os melhores scores
print("Melhores configurações do Lasso:")
print(lasso_grid.best_params_)
print("Melhor score do Lasso:")
print(lasso_grid.best_score_)

print("\nMelhores configurações do Ridge:")
print(ridge_grid.best_params_)
print("Melhor score do Ridge:")
print(ridge_grid.best_score_)
