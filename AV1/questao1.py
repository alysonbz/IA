#importe as bibliotecas necessárias
import pandas as pd



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
print("\n colunas relevantes")
df1_relevantes=df1[['MinTemp',
            'MaxTemp',
            'RainTomorrow',
            'RainToday',]]


#Print o dataframe final e mostre a distribuição de classes que você deve classificar
print(df1_relevantes)
#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário
df1_relevantes['RainTomorrow'].replace(['Yes', 'No'],
                        [0, 1], inplace=True)
df1_relevantes['RainToday'].replace(['Yes', 'No'],
                        [0, 1], inplace=True)
print("dataser modificado")
print(df1_relevantes)
#Salve o dataset atualizado se houver modificações.
df1_relevantes.to_csv("df1_final.csv")
