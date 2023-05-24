#importe as bibliotecas necessárias

import pandas as pd


## Carregue o dataset definido para você

Samsung = pd.read_csv(r"C:\Users\Aluno\Downloads\archive\Samsung Electronics.csv")

print(Samsung)

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.

Samsung_num = Samsung.isnull().sum()

print(Samsung_num)


#Print o dataframe final e mostre a distribuição de classes que você deve classificar

print(Samsung['Date'].value_counts())
print(Samsung['Open'].value_counts())
print(Samsung['High'].value_counts())
print(Samsung['Low'].value_counts())
print(Samsung['Close'].value_counts())
print(Samsung['Adj Close'].value_counts())
print(Samsung['Volume'].value_counts())

#Salve o dataset atualizado se houver modificações.

Samsung.to_csv('Samsung_ajustado', index=False)

#target mais relevante para a regressão

#UNIDADE 4