#importe as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import load_sales_clean_dataset
from sklearn.linear_model import Lasso

## Carregue o dataset definido para você
df = pd.read_csv(r"C:\Users\LAB1_00\Desktop\SAVIO\IA\AV2\Ubisoft.csv")
'''print(df)
print(df.columns)
print(df.info())'''
# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(df.isnull().sum())
df_limpo = df.dropna()
print(df_limpo.isnull().sum())

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
X = df_limpo.drop(["sales","influencer"], axis=1)
y = df.limpo["open"].values
sales_columns = X.columns


#Print o dataframe final e mostre a distribuição de classes que você deve classificar



#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário



#Salve o dataset atualizado se houver modificações.
