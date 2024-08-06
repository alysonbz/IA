#importe as bibliotecas necessárias
import pandas as pd
from sklearn.preprocessing import LabelEncoder

## Carregue o dataset definido para você
drug200= pd.read_csv('dataset/drug200.csv')
print(drug200.head())


# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(drug200.isna().sum())




# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
print(drug200, "\n Todas parecem ser relevantes!")



#Print o dataframe final e mostre a distribuição de classes que você deve classificar
print(drug200)

print("Distribuição da classe Drug:")
class_dist = drug200
print(class_dist['Drug'].value_counts(), '\n','\n')



#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário
print(drug200.info())

# Colunas do tipo object
columns_to_encode = ['Sex', 'BP', 'Cholesterol', 'Drug']

# Convertendo com LabelEncoder
le = LabelEncoder()
for column in columns_to_encode:
   drug200[column] = le.fit_transform(drug200[column])

print("Novos tipos:")
print(drug200.info())

#Salve o dataset atualizado se houver modificações.
drug200.to_csv('dataset/drug200_new.csv', index=False)