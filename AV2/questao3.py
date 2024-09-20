# Importando bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Carregando o dataset
file_path = 'C:\\Users\\Neto\\Downloads\\IA\\AV2\\Dataset\\Sample - Superstore.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Selecionando apenas os atributos numéricos relevantes para "Sales"
# Incluindo "Profit", "Quantity" e removendo "Row ID", "Postal Code", "Discount"
X = df[['Profit', 'Quantity']]  # Atributos mais relevantes
y = df['Sales']  # Atributo alvo

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definindo os hiperparâmetros para GridSearchCV
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}

# Regressão Lasso
lasso = Lasso()
lasso_cv = GridSearchCV(lasso, param_grid, cv=5)
lasso_cv.fit(X_train, y_train)

# Regressão Ridge
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid, cv=5)
ridge_cv.fit(X_train, y_train)

# Avaliando os modelos
lasso_best = lasso_cv.best_estimator_
ridge_best = ridge_cv.best_estimator_

y_pred_lasso = lasso_best.predict(X_test)
y_pred_ridge = ridge_best.predict(X_test)

# Calculando os scores para ambos os modelos
lasso_score = lasso_cv.best_score_
ridge_score = ridge_cv.best_score_

# Exibindo os melhores hiperparâmetros e scores
print("Melhor configuração para Lasso:", lasso_cv.best_params_)
print("Melhor score para Lasso (Cross-Validation):", lasso_score)

print("Melhor configuração para Ridge:", ridge_cv.best_params_)
print("Melhor score para Ridge (Cross-Validation):", ridge_score)

# Avaliação dos modelos no conjunto de teste
print("R² para Lasso no conjunto de teste:", r2_score(y_test, y_pred_lasso))
print("R² para Ridge no conjunto de teste:", r2_score(y_test, y_pred_ridge))

print("RMSE para Lasso no conjunto de teste:", mean_squared_error(y_test, y_pred_lasso, squared=False))
print("RMSE para Ridge no conjunto de teste:", mean_squared_error(y_test, y_pred_ridge, squared=False))