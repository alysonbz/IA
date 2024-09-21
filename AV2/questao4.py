"""
Utilizando kfold e cross-validation faça uma regressão linear utilizando os mesmos atributos
definidos na questão 3. Obs: Com os resultados obtidos na questão 3 e da questão 4 faça uma
comparação entre os desempenhos. Escolha o regressor adequado e informe o motivo da escolha.
Discuta sobre as limitações e acertos encontrados.
"""
from src.utils import load_laptopPrice_dataset_cleaned
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

# Carregar os dados
laptop = load_laptopPrice_dataset_cleaned()

# Selecionar os atributos relevantes
relevant_columns = ['graphic_card_gb', 'ram_type', 'processor_name', 'Touchscreen']
X_relevant = laptop[relevant_columns]
y = laptop['Price']

# Definir o modelo de regressão linear
model_cv = LinearRegression()

# Configurar K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Executar a validação cruzada
cv_scores = cross_val_score(model_cv, X_relevant, y, cv=kf, scoring='neg_mean_squared_error')

# Converter os scores para valores positivos
mse_scores = -cv_scores
mean_mse = mse_scores.mean()
mean_rmse = np.sqrt(mean_mse)
mean_r_squared = 1 - (mean_mse / np.var(y))

# Mostrar resultados
print(f"Média do MSE (K-Fold): {mean_mse}")
print(f"Média do RMSE (K-Fold): {mean_rmse}")
print(f"Média do R² (K-Fold): {mean_r_squared}")