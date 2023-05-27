#importe as bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder


## Carregue o dataset definido para você
df= pd.read_csv(r"C:\Users\ruanr\Downloads\archive (1)\bodyfat.csv")
df.info()



# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print("\n Verificação da existência de células vaizas ou Nan")
print(df.isna().sum())



# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
print("\n colunas relevantes")
df_final = df[['Density',
                          'BodyFat',
                          'Age',
                          'Weight',
                         'Height',]]


#Print o dataframe final e mostre a distribuição de classes que você deve classificar
print(df_final)
# Create X and y arrays
X = df_final.drop(["BodyFat"], axis=1).select_dtypes(exclude=["object"])
y = df_final["BodyFat"].values
sales_columns = X.columns
# Instantiate a lasso regression model
lasso = Lasso(alpha=0.3)

# Compute and print the coefficients
lasso_coef = lasso.fit(X,y).coef_
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)

plt.xticks(rotation=45)
plt.show()


#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário
# Criar uma instância do LabelEncoder
#label_encoder = LabelEncoder()
# Aplicar o LabelEncoder às colunas categóricas
#for coluna in bodyfat_ajust:
    #if bodyfat_ajust[coluna].dtype == 'object':
        #bodyfat_ajust.loc[:, coluna] = label_encoder.fit_transform(bodyfat_ajust[coluna])
#atributo mais relevante
#X = bodyfat_ajust.drop(['BodyFat'], axis=1)
#y = bodyfat_ajust['BodyFat'].values
#y = bodyfat_ajust['BodyFat'].values
#bodyfat_ajust_columns = X.columns
''# Instantiate a lasso regression model
#lasso = Lasso(alpha=0.1)
# Instantiate a lasso regression model
#lasso = Lasso(alpha=0.3)

# Compute and print the coefficients
#lasso_coef = lasso.fit(X, y).coef_
#print(lasso_coef)
#plt.bar(bodyfat_ajust_columns, lasso_coef)
#plt.xticks(rotation=45)
#plt.show()


df_final.to_csv("df_final.csv")

#Salve o dataset atualizado se houver modificações.
#bodyfat_ajust.to_csv("bodyfat_final.csv")