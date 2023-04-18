#importe as bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


#Carregue o dataset definido para você
df1=pd.read_csv(r'C:\Users\ruanr\IA\AV1\dataset\weather.csv')
print(df1.head)
print(df1.shape)
print(df1.info)
print(df1.describe)

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(df1.isnull().sum())
df1 = df1.dropna()
print(df1.isnull().sum())





# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
relevantes=['MinTemp',
            'MaxTemp',
            'RainTomorrow',
            'RainToday',
            'WindGustDir',]
df2 =pd.DataFrame(relevantes,columns=['relevantes'])


#Print o dataframe final e mostre a distribuição de classes que você deve classificar
print(df2)

#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário
print(df1.info())


#Salve o dataset atualizado se houver modificações.
