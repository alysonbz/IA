import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.utils import load_lenovo_share_prices  # Importar a função de utils.py

# 1. Carregar o dataset
lenovo_df = load_lenovo_share_prices()

# 2. Remover linhas com valores ausentes
lenovo_df = lenovo_df.dropna(subset=['Open', 'High', 'Low', 'Volume', 'Close'])

# 3. Selecionar os atributos relevantes para regressão
X = lenovo_df[['Open', 'High', 'Low', 'Volume']]  # Atributos relevantes
y = lenovo_df['Close']  # Alvo

# 4. Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Criar pipelines para Lasso e Ridge com normalização
lasso_pipeline = make_pipeline(StandardScaler(), Lasso(max_iter=5000))  # Aumentando o número de iterações no Lasso
ridge_pipeline = make_pipeline(StandardScaler(), Ridge())

# 6. Definir os parâmetros para GridSearchCV
param_grid = {'lasso__alpha': [0.001, 0.01, 0.1, 1, 10, 100], 'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# 7. Configurar o modelo Lasso com GridSearchCV
lasso_cv = GridSearchCV(lasso_pipeline, {'lasso__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}, cv=5, scoring='neg_mean_squared_error')
lasso_cv.fit(X_train, y_train)

# 8. Configurar o modelo Ridge com GridSearchCV
ridge_cv = GridSearchCV(ridge_pipeline, {'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train, y_train)

# 9. Melhor configuração e score para Lasso
best_lasso_alpha = lasso_cv.best_params_['lasso__alpha']
best_lasso_score = -lasso_cv.best_score_

# 10. Melhor configuração e score para Ridge
best_ridge_alpha = ridge_cv.best_params_['ridge__alpha']
best_ridge_score = -ridge_cv.best_score_

# 11. Imprimir os resultados
print(f"Melhor configuração para Lasso: alpha = {best_lasso_alpha}")
print(f"Melhor score (MSE) para Lasso: {best_lasso_score}")
print(f"Melhor configuração para Ridge: alpha = {best_ridge_alpha}")
print(f"Melhor score (MSE) para Ridge: {best_ridge_score}")

# 12. Avaliação no conjunto de teste
y_pred_lasso = lasso_cv.best_estimator_.predict(X_test)
y_pred_ridge = ridge_cv.best_estimator_.predict(X_test)

# 13. Cálculo do MSE para os conjuntos de teste
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

print(f"MSE no conjunto de teste para Lasso: {mse_lasso}")
print(f"MSE no conjunto de teste para Ridge: {mse_ridge}")
