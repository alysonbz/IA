import numpy as np
from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression

# Importa os módulos necessários.
from sklearn.model_selection import KFold, cross_val_score

sales_df = load_sales_clean_dataset()

# Cria matrizes X e y.
X = sales_df["radio"].values.reshape(-1, 1)
y = sales_df["sales"].values

# Cria um objeto KFold.
kf = KFold(n_splits=6, shuffle=True, random_state=5)

reg = LinearRegression()

# Calcula pontuações de validação cruzada de 6 vezes.
cv_scores = cross_val_score(reg, X, y, cv=kf)

# Impressão do cv_scores.
print(cv_scores)

# Imprime a médias.
print(np.mean(cv_scores))

# Imprime o desvio padrão.
print(np.std(cv_scores))
