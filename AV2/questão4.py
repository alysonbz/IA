import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Carregar o dataset
data = pd.read_csv(r'C:\Users\eryka\Downloads\archive\Samsung Electronics.csv')

# Selecionar os atributos relevantes
relevant_attributes = ['High', 'Low', 'Close']  # Exemplo, substitua pelos atributos relevantes identificados

# Filtrar o DataFrame com os atributos relevantes
data_relevant = data[relevant_attributes]

X = data[['Open', 'Low', 'Volume']].values
y = data[['Close', 'High']].values


# Definir os atributos de entrada (X) e o atributo alvo (y)
X = data_relevant.drop(['Close', 'High'], axis=1).values
y = data_relevant[['Close', 'High']].values

# Criar um objeto de regressão linear
regressor = LinearRegression()

# Executar k-fold cross-validation com 5 folds
scores = cross_val_score(regressor, X, y, cv=5)

# Imprimir os scores de desempenho para cada fold
print("Scores de desempenho:", scores)
