#importe as bibliotecas necessárias
import pandas as pd

## Carregue o dataset definido para você
choco = pd.read_csv(r"C:\Users\Aluno\Desktop\savio\IA\AV1\dataset\flavors_of_cacao.csv")
#print(choco.head(n=10))
#print(choco.describe())
#print(choco.info)

#Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(choco.isnull().sum())
'''
volunteer_subset = volunteer_cols.dropna(subset=['category_desc'])
choco = choco.drop(['Latitude', 'Longitude'], axis=1)

choco_new = choco.dropna()'''

#Verifique quais colunas são as mais relevantes e crie um novo dataframe.



#Print o dataframe final e mostre a distribuição de classes que você deve classificar



#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário



#Salve o dataset atualizado se houver modificações.


