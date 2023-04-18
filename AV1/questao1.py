#importe as bibliotecas necessárias
import pandas as pd

## Carregue o dataset definido para você
df = pd.read_csv(r'C:\Users\LAB1_00\Documents\GD\IA\AV1\dataset\dataset__binary.csv')

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(df.isna().sum())
print(df.isnull().sum())

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
print(df.info())
print(df.head())
print(df[["var_1", "var_2"]].value_counts())

#Print o dataframe final e mostre a distribuição de classes que você deve classificar



#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário



#Salve o dataset atualizado se houver modifcações.i