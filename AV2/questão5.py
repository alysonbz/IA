#bibliotecas
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold


emissao_df = pd.read_csv('dados.csv.csv')

# Separar os atributos de entrada (X) e o atributo alvo (y)
X = emissao_df.drop(["CO2 Emissions(g/km)"], axis=1).select_dtypes(exclude=["object"])
y = emissao_df['CO2 Emissions(g/km)']


# Inicializar o modelo de regressão linear
regressor = LinearRegression()

# Inicializar o objeto KFold para validação cruzada
kf = KFold(n_splits=6, shuffle=True, random_state=42)

# Realizar a validação cruzada e obter os scores
cv_scores = cross_val_score(regressor, X, y, cv=kf)

# Imprimir os scores para cada fold
print("Scores para cada fold:")
print(cv_scores)

# Calcular a média dos scores
mean_score = cv_scores.mean()
print("Média dos scores:", mean_score)