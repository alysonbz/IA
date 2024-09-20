# Importando bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Carregando o dataset
file_path = 'C:\\Users\\Neto\\Downloads\\IA\\AV2\\Dataset\\Sample - Superstore.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Selecionando os atributos
X = df[['Profit', 'Quantity']]  # Atributos relevantes
y = df['Sales']  # Atributo alvo

# Inicializando
linear_model = LinearRegression()

# Definindo K-Fold Cross-Validation (usaremos 5 divisões)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Avaliando o modelo utilizando K-Fold Cross-Validation (MSE e R²)
mse_scores = cross_val_score(linear_model, X, y, scoring='neg_mean_squared_error', cv=kf)
r2_scores = cross_val_score(linear_model, X, y, scoring='r2', cv=kf)

# Convertendo
mse_scores_positive = -mse_scores
rmse_scores = mse_scores_positive ** 0.5

# Exibindo os resultados de Cross-Validation
mse_cv_mean = mse_scores_positive.mean()
rmse_cv_mean = rmse_scores.mean()
r2_cv_mean = r2_scores.mean()

# Comparando os resultados com os valores de Lasso e Ridge da questão anterior
lasso_r2 = 0.29  # Resultado fictício
lasso_rmse = 510  # Resultado fictício
ridge_r2 = 0.31  # Resultado fictício
ridge_rmse = 505  # Resultado fictício

# Comparação dos resultados
comparacao_resultados = pd.DataFrame({
    "Modelo": ["Regressão Linear", "Lasso", "Ridge"],
    "R²": [r2_cv_mean, lasso_r2, ridge_r2],
    "RMSE": [rmse_cv_mean, lasso_rmse, ridge_rmse]
})

# Exibindo resultados
print(comparacao_resultados)
