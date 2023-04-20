#importe as bibliotecas necessárias
import pandas as pd
core= pd.read_csv('dataset/heart_desease.csv')

## Carregue o dataset definido para você
print(core.to_string())

# Verifique se existem celulas vazias ou Nan.
# Se existir, excluir e criar um novo dataframe.
print(core.isna().sum(), '\n')

# Verifique quais colunas são as mais relevantes
# e crie um novo dataframe.


# Print o dataframe final e mostre a distribuição
# de classes que você deve classificar
print(core)
print(core['Age'].value_counts(),'\n','\n')
print(core['Sex'].value_counts(),'\n','\n')
print(core['ChestPainType'].value_counts(),'\n','\n')
print(core['RestingBP'].value_counts(),'\n','\n')
print(core['Cholesterol'].value_counts(),'\n','\n')
print(core['FastingBS'].value_counts(),'\n','\n')
print(core['ChestPainType'].value_counts(),'\n','\n')
print(core['RestingECG'].value_counts(),'\n','\n')
print(core['MaxHR'].value_counts(),'\n','\n')
print(core['ExerciseAngina'].value_counts(),'\n','\n')
print(core['Oldpeak'].value_counts(),'\n','\n')
print(core['ST_Slope'].value_counts(),'\n','\n')
print(core['HeartDisease'].value_counts(),'\n','\n')

#observe se a coluna de classes precisa ser renomeada
#para atributos numéricos, realize a conversão, se necessário
core['Sexo'] = core['Sex'].replace({'M': 0, 'F': 1})
print(core['Sexo'])
core['ChestPain']=core['ChestPainType'].replace({'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3})
print(core['ChestPain'])
core['RestECG']=core['RestingECG'].replace({'Normal':0, 'ST':1, 'LVH':2})
print(core['RestECG'])
core['ExerciseAng']=core['ExerciseAngina'].replace({'N':0, 'Y':1})
print(core['ExerciseAng'])
core['SlopST']=core['ST_Slope'].replace({'Up':0, 'Flat':1, 'Down':3})
print(core['SlopST'])

core=core.drop('Sex', axis=1)
core=core.drop('ChestPainType', axis=1)
core=core.drop('RestingECG', axis=1)
core=core.drop('ExerciseAngina', axis=1)
core=core.drop('ST_Slope', axis=1)


#Salve o dataset atualizado se houver modificações.
#heart_desease_ajustado = core
hda=core
print(hda)
hda.to_csv=('dataset/heart_desease.csv')

