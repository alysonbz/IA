#importe as bibliotecas necessárias
import pandas as pd

## Carregue o dataset definido para você

df = pd.read_csv(r'C:\Users\Guilherme\Documents\G\IA\AV1\dataset\dataset__binary.csv')

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print("na: \n", df.isna().sum())
print("null: \n",df.isnull().sum())

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
print("head: \n", df.head())
print("value counts: \n", df[["target"]].value_counts())

#Print o dataframe final e mostre a distribuição de classes que você deve classificar
print("dataframe:\n", df)


#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário
# não necessário


#Salve o dataset atualizado se houver modifcações.
# não necessário