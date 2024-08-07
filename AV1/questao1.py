# importe as bibliotecas necessárias
import pandas as pd

## Carregue o dataset definido para você
diabetes = pd.read_csv('AV1/dataset/diabetes.csv')
print(diabetes.head())

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
diabetes_limpo = diabetes.dropna() 

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
diabetes_relevante = diabetes_limpo[['Outcome', 'Age', 'BloodPressure', 'Glucose']]

# Print o dataframe final e mostre a distribuição de classes que você deve classificar
print(diabetes_relevante.head())
print(diabetes_relevante['Outcome'].value_counts())

# Observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário
diabetes_relevante['Outcome'] = diabetes_relevante['Outcome'].replace({1: 'Sim', 0: 'Não'})

# Salve o dataset atualizado se houver modificações.
diabetes_relevante.to_csv('diabetes_atualizado.csv', index=False) 
