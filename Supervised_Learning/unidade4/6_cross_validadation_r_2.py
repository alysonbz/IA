from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

# Carregar o dataset
sales_df = load_sales_clean_dataset()

# Criar X e y arrays
X = sales_df["radio"].values.reshape(-1, 1)
y = sales_df["sales"].values

# Criar um objeto KFold com 6 divisões e embaralhamento
kf = KFold(n_splits=6, shuffle=True, random_state=5)

# Inicializar o modelo de regressão
reg = LinearRegression()

# Computar os escores de validação cruzada com 6-fold
cv_scores = cross_val_score(reg, X, y, cv=kf)

# Imprimir os escores de validação cruzada
print("CV Scores: {}".format(cv_scores))

# Imprimir a média dos escores
print("Mean CV Score: {}".format(np.mean(cv_scores)))

# Imprimir o desvio padrão dos escores
print("Standard Deviation of CV Scores: {}".format(np.std(cv_scores)))
