#importe as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt

## Carregue o dataset definido para você
flavors_of_cacao = pd.read_csv(r"C:\Users\Aluno\Downloads\Savio\IA\AV1\dataset\flavors_of_cacao.csv")
'''print(flavors_of_cacao.head(n=10))
print(flavors_of_cacao.info())
'''
#Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
#print(flavors_of_cacao.isnull().sum())
flavors_of_cacao = flavors_of_cacao.dropna()
flavors_of_cacao_new = flavors_of_cacao.drop(['Company \r\n(Maker-if known)', 'REF', 'Review\r\nDate', 'Company\r\nLocation', 'Bean\r\nType'], axis=1)
'''print(flavors_of_cacao_new)'''

#Verifique quais colunas são as mais relevantes e crie um novo dataframe.

flavors_of_cacao_ajustado = flavors_of_cacao_new

#Print o dataframe final e mostre a distribuição de classes que você deve classificar

print(flavors_of_cacao_ajustado.columns)
a = flavors_of_cacao_ajustado['Cocoa\r\nPercent'].value_counts()
b = flavors_of_cacao_ajustado['Rating'].value_counts()
c = flavors_of_cacao_ajustado['Broad Bean\r\nOrigin'].value_counts()
#print(a)
#print(b)
#print(C)

#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário



#Salve o dataset atualizado se houver modificações.

