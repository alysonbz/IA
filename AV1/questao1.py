#importe as bibliotecas necessárias
import pandas as pd

## Carregue o dataset definido para você
cancer = pd.read_csv('breast-cancer.csv')
print(cancer.head())

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(cancer.isnull().sum())  #quantos null
print(cancer.isna().sum())  #quantos na

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
print(cancer.columns)

corr = cancer.corr()
print(corr)

#cancer_new =


#Print o dataframe final e mostre a distribuição de classes que você deve classificar



#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário



#Salve o dataset atualizado se houver modificações.