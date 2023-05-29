### Questão 1

#Verifique qual atributo será o alvo para regressão no seu dataset
#e faça uma análise de qual atributo é mais relevante para realizar a regressão do alvo escolhido.
#Lembre de comprovar via gráfico.
#Obs: Registrar na seção de resultados a análise realizada e discutir sobre o resultado encontrado.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

df = pd.read_csv(r'C:\Users\Guilherme\Documents\G\IA\AV2\Sample - Superstore.csv', encoding='latin-1')

print(df.columns)
#print(df.head())
#print(df.describe())
print("Info:", df.info())

#X = df.drop(["Sales"], axis=1).select_dtypes(exclude="object")
#y = df["Sales"].values
#sales_columns = df.drop("Sales", axis=1).columns0
#lasso = Lasso(alpha=0.1)
#lasso_coef = lasso.fit(X,y).coef_
#print(lasso_coef)
#plt.bar(sales_columns, lasso_coef)
#plt.xticks(rotation=45)
#plt.show()


# Separar os atributos e o alvo
X = df.drop(['Sales'], axis=1)
y = df['Sales']

# Instanciar o modelo Lasso
lasso = Lasso(alpha=0.1)

# Ajustar o modelo aos dados
lasso.fit(X, y)

# Obter os coeficientes
lasso_coef = pd.Series(lasso.coef_, index=X.columns)

# Ordenar os coeficientes em ordem decrescente de importância
sorted_coef = lasso_coef.abs().sort_values(ascending=False)

# Exibir os atributos mais importantes
print(sorted_coef)
