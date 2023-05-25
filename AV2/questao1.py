#importando as bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
#import data set
df = pd.read_csv(r"C:\Users\Aluno\Desktop\savio\IA\AV2\Ubisoft.csv")
#print(df)
print(df.columns)
#Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
'''print(df.isnull().sum())'''
df = df.dropna()
'''print(df.isnull().sum())'''

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
#df = df.drop(['Date'], axis=1)

# Create X and y arrays
X = df.drop(["Open","Date"], axis=1)
y = df["Open"].values
sales_columns = X.columns

# Instantiate a lasso regression model
lasso = Lasso(alpha=0.3)

# Compute and print the coefficients
lasso_coef = lasso.fit(X,y).coef_
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()


# Calcular a matriz de correlação
correlation_matrix = df.corr()

# Exibir a matriz de correlação
print(correlation_matrix)

# Plotar um gráfico de calor para visualizar a matriz de correlação
plt.figure(figsize=(10, 8))
plt.title('Matriz de Correlação')
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(df.columns)), df.columns, rotation=45)
plt.yticks(range(len(df.columns)), df.columns)
plt.show()