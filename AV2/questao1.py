### Questão 1

import pandas as pd
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\Guilherme\Documents\G\IA\AV2\Sample - Superstore.csv', encoding='latin-1')

#Verifique qual atributo será o alvo para regressão no seu dataset
print(df.columns)
print(df.head())
print(df.info())

target = 'Profit' #Lucro

#Faça uma análise de qual atributo é mais relevante para realizar a regressão do alvo escolhido.
X = df.drop(target, axis=1).select_dtypes(exclude="object")
y = df[target].values
sales_columns = X.columns
lasso = Lasso(alpha=0.1)
lasso.fit(X,y)
lasso_coef = pd.Series(lasso.coef_, index=X.columns)
print('Importância dos atributos:\n',lasso_coef.sort_values(ascending=False))

#Comprove via gráfico.
plt.bar(X.columns, lasso_coef, color="#ff6f9c")
plt.xlabel('Atributos',fontweight='bold')
plt.ylabel('Importância',fontweight='bold')
plt.title('Importância dos atributos na regressão de Profit',fontweight='bold')
plt.show()

#Obs: Registrar na seção de resultados a análise realizada e discutir sobre o resultado encontrado.
