import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

#import dataset
df = pd.read_csv(r"C:\Users\Eduardo\Desktop\Prova_IA_2\Lenovo_share_prices.csv")

#print(df.columns)
print(df.head(5))

#Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(df.isnull().sum())
df2 = df.dropna()
print(df2.isnull().sum())

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
# Create X and y arrays
X = df2.drop(["Open",'Date'], axis=1)
y = df2["Open"].values
sales_columns = X.columns

# Instantiate a lasso regression model
lasso = Lasso(alpha=0.3)

# Compute and print the coefficients
lasso_coef = lasso.fit(X,y).coef_
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()

# Removendo a coluna Date e nomeando o novo df
df2 = df2.drop(['Date'], axis=1)
Lenovo = pd.DataFrame(df)
df2.to_csv('lenovo.csv', index=False)