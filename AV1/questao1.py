#importe as bibliotecas necessárias
import pandas as pd

## Carregue o dataset definido para você
def load_binary_dataset():
    return pd.read_csv(r'C:\Users\LAB1_00\Documents\GD\IA\AV1\dataset\dataset__binary.csv')
df = load_binary_dataset()

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print("na: \n", df.isna().sum())
print("null: \n",df.isnull().sum())

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
print("head: \n", df.head())
print("value counts: \n", df[["target"]].value_counts())

#Print o dataframe final e mostre a distribuição de classes que você deve classificar
print("dataframe:\n", df)


#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário
# não necessario


#Salve o dataset atualizado se houver modifcações.i
# não necessario