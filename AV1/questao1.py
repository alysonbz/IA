#importe as bibliotecas necessárias

import pandas as pd

## Carregue o dataset definido para você

airline_brute = pd.read_csv(r"C:\Users\Aluno\Documents\444\IA\AV1\dataset\airline_satisfaction.csv")
print(airline_brute)

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.

airline_processing = airline_brute.isnull().sum()
print(airline_processing)  #encontramos apenas a coluna "Arrival Delay in Minutes"

airline_modification = airline_processing.drop(["Arrival Delay in Minutes"])
print(airline_modification)

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.

print(airline_modification.info()) #fazendo uma analise mais manual das colunas, percebemos irrelevancia nas colunas Id, idade e Hora de partida/chegada conveniente
airline_fake = airline_modification.drop(["id", "Age", "Departure/Arrival time convenient"]) # 3 retiradas no total
print(airline_fake)

#Print o dataframe final e mostre a distribuição de classes que você deve classificar


#print(airline_fake.info())
#print(airline_brute['Gender'].value_counts())
#print(airline['Class'].value_counts())
#print(airline['Type of Travel'].value_counts())
#print(airline['satisfaction'].value_counts())


#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário



#Salve o dataset atualizado se houver modificações.