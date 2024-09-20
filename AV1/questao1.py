<<<<<<< HEAD
# Importe as bibliotecas necessárias.
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Carregue o dataset definido para você.
healthcare_data= pd.read_csv('dataset/healthcare-dataset-stroke-data.csv')

# Mostrar as primeiras linhas do dataset
print(healthcare_data.head())


# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(healthcare_data.isna().sum())
healthcare_data = healthcare_data.dropna(axis=0)
print(healthcare_data.isna().sum())


# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
print("Coluna 'id' irrelevante para a análise. Sendo deletada!")
healthcare_data_new = healthcare_data.drop("id", axis=1)

print("Apresentando dataset novo:")
print(healthcare_data_new)


# Print o dataframe final e mostre a distribuição de classes que você deve classificar.
print(healthcare_data_new)

print("Distribuição da classe stroke:")
class_dist = healthcare_data_new
print(class_dist['stroke'].value_counts(), '\n','\n')


# Observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário.
print("Tipos")
print(healthcare_data_new.dtypes)

# Lista de colunas para codificar
columns_to_encode = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# Aplica o LabelEncoder a cada coluna
le = LabelEncoder()

for column in columns_to_encode:
   healthcare_data_new[column] = le.fit_transform(healthcare_data_new[column])
print()

print("Convertida")
print(healthcare_data_new.info())



# Salve o dataset atualizado se houver modificações.
healthcare_data_new.to_csv('dataset/healthcare-dataset-stroke-data-new.csv', index=False)
=======
#importe as bibliotecas necessárias


## Carregue o dataset definido para você



# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.



# Verifique quais colunas são as mais relevantes e crie um novo dataframe.



#Print o dataframe final e mostre a distribuição de classes que você deve classificar



#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário



#Salve o dataset atualizado se houver modificações.
>>>>>>> 97e940d2647336b47c19a5910c43581bbe81a5bb
