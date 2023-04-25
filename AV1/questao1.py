#importe as bibliotecas necessárias
import pandas as pd

## Carregue o dataset definido para você
print('Carregue o dataset definido para você')
cancer = pd.read_csv('breast-cancer.csv')  #código que roda o dataset
print(cancer.head())  #mostra o dataframe


# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print('Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.')
print(cancer.isnull().sum())  #quantos null tem em cada coluna
print(cancer.isna().sum())  #quantos na tem em cada coluna

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
print('Verifique quais colunas são as mais relevantes e crie um novo dataframe.')

print(cancer.columns)  #mostra as colunas do dataframe

cancer_relevancia = cancer[['id', 'diagnosis', 'radius_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']] #cria um novo dataframe
cancer = pd.DataFrame(cancer[['id', 'diagnosis', 'radius_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']])



# Print o dataframe final e mostre a distribuição de classes que você deve classificar
print('Print o dataframe final e mostre a distribuição de classes que você deve classificar')

print(cancer_relevancia['id'].value_counts(),'\n','\n')  #faz a contagem de cada valor unico
print(cancer_relevancia['diagnosis'].value_counts(),'\n','\n')
print(cancer_relevancia['perimeter_mean'].value_counts(),'\n','\n')
print(cancer_relevancia['radius_mean'].value_counts(),'\n','\n')
print(cancer_relevancia['area_mean'].value_counts(),'\n','\n')
print(cancer_relevancia['smoothness_mean'].value_counts(),'\n','\n')


#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário

cancer_relevancia['diagnosis'].replace(['B', 'M'],
                        [0, 1], inplace=True)  #essa função serve para alterar o B pra 0 e M pra 1


#Salve o dataset atualizado se houver modificações.
print('Salve o dataset atualizado se houver modificações.')
print(cancer_relevancia)

cancer_relevancia.to_csv("dados_preprocessados.csv", index=False)