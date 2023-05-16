#importe as bibliotecas necessárias
import pandas as pd
import numpy as np

## Carregue o dataset definido para você
df = pd.read_csv(r"C:\Users\LAB1_00\Desktop\kelvincbl\IA\AV1\dataset\waterQuality1.csv")
print(df.info())

# Verifique se existem colunas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(df.isna().sum())

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
df = df['']


#Print o dataframe final e mostre a distribuição de classes que você deve classificar
print(df['Potability'].value_counts())


#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário
df.rename(columns={'Potability': 'Class'}, inplace=True)


#Salve o dataset atualizado se houver modificações.
df.to_csv('water_quality_ajustado.csv', index=False)
