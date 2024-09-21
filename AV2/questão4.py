import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
# Carregar o dataset
clean_dataset = pd.read_csv('./dataset/Clean_Dataset.csv')

clean_dataset["class"] = clean_dataset["class"].replace("Economy", 0)
clean_dataset["class"] = clean_dataset["class"].replace("Business", 1)

X = clean_dataset[['class']].values
y = clean_dataset['price'].values

# Criar o modelo de regressão linear
linear_model = LinearRegression()

# Realizar a validação cruzada com K-Fold (5 divisões)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_linear = cross_val_score(linear_model, X, y, cv=kf, scoring='r2')

# Calcular a média e o desvio padrão dos scores R²
mean_cv_score_linear = np.mean(cv_scores_linear)
std_cv_score_linear = np.std(cv_scores_linear)

print("Regressão Linear:")
print("Score médio (R²) da regressão linear com K-Fold:", mean_cv_score_linear)
print("Desvio padrão dos scores R²:", std_cv_score_linear)

# Criar os modelos Lasso e Ridge
lasso_model = Lasso(alpha=0.0001)  # Valor de alpha obtido anteriormente
ridge_model = Ridge(alpha=0.0001)  # Valor de alpha obtido anteriormente

# Realizar a validação cruzada com K-Fold para Lasso
cv_scores_lasso = cross_val_score(lasso_model, X, y, cv=kf, scoring='r2')
mean_cv_score_lasso = np.mean(cv_scores_lasso)
std_cv_score_lasso = np.std(cv_scores_lasso)

# Realizar a validação cruzada com K-Fold para Ridge
cv_scores_ridge = cross_val_score(ridge_model, X, y, cv=kf, scoring='r2')
mean_cv_score_ridge = np.mean(cv_scores_ridge)
std_cv_score_ridge = np.std(cv_scores_ridge)

# Comparação dos resultados
print("\nLasso:")
print("Score médio (R²):", mean_cv_score_lasso)
print("Desvio padrão dos scores R²:", std_cv_score_lasso)

print("\nRidge:")
print("Score médio (R²):", mean_cv_score_ridge)
print("Desvio padrão dos scores R²:", std_cv_score_ridge)

# Comparação dos resultados dos três modelos
print("\nComparação dos Modelos:")
print("Regressão Linear - Média R²:", mean_cv_score_linear, "Desvio Padrão:", std_cv_score_linear)
print("Lasso - Média R²:", mean_cv_score_lasso, "Desvio Padrão:", std_cv_score_lasso)
print("Ridge - Média R²:", mean_cv_score_ridge, "Desvio Padrão:", std_cv_score_ridge)

# Escolha do melhor regressor baseado nos resultados:
if mean_cv_score_linear >= mean_cv_score_lasso and mean_cv_score_linear >= mean_cv_score_ridge:
    print("\nRegressão Linear é o modelo mais adequado.")
elif mean_cv_score_lasso > mean_cv_score_linear and mean_cv_score_lasso >= mean_cv_score_ridge:
    print("\nLasso é o modelo mais adequado.")
else:
    print("\nRidge é o modelo mais adequado.")