#importe as bibliotecas necessárias
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd

## Carregue o dataset definido para você
df = pd.read_csv(r"C:\Users\Aluno\Desktop\savio\IA\AV1\dataset\flavors_of_cacao.csv")
'''print(df.head(n=10))'''
print(df.info())
print(df.columns)
#Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
#print(df.isnull().sum())
df = df.dropna()
#print(df.columns)
df = df.drop(['Company \r\n(Maker-if known)', 'Specific Bean Origin\r\nor Bar Name', 'Company\r\nLocation','REF','Review\r\nDate', 'Bean\r\nType', 'Broad Bean\r\nOrigin'], axis=1)
print(df.head(n=10))

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
print(df.columns)
df_new = pd.DataFrame(df)

#Print o dataframe final e mostre a distribuição de classes que você deve classificar

print(df_new['Rating'].value_counts())
print(df_new['Cocoa\r\nPercent'].value_counts())
#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário

df_new['Cocoa\r\nPercent'] = df_new['Cocoa\r\nPercent'].str.replace('%', '')
df_new['Cocoa\r\nPercent'] = pd.to_numeric(df_new['Cocoa\r\nPercent'])


#Salve o dataset atualizado se houver modificações.

df_new.to_csv('flavors_of_cacao_ajustado.csv')

print(df_new.info())