#importe as bibliotecas necessárias

import pandas as pd

## Carregue o dataset definido para você

airline_brute = pd.read_csv(r"C:\Users\Aluno\Desktop\444\IA\AV1\dataset\airline_satisfaction.csv")
print(airline_brute)

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.

airline_processing = airline_brute.isnull().sum()
print(airline_processing)  #encontramos apenas a coluna "Arrival Delay in Minutes"

airline_modification = airline_processing.drop(["Arrival Delay in Minutes"])
print(airline_modification)

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.

print(airline_modification.info())
airline = airline_modification.drop(["id", "Age", "Gender", "Flight Distance", "Departure/Arrival time convenient", "Gate location", "Customer Type", "Departure Delay in Minutes" ]) # 3 retiradas no total
#print(airline)

#Print o dataframe final e mostre a distribuição de classes que você deve classificar

airline = pd.DataFrame(airline)
print(airline)

print(airline.info())
print(airline_brute['Type of Travel'].value_counts())
print(airline_brute['Class'].value_counts())
print(airline_brute['Inflight wifi service'].value_counts())
print(airline_brute['Ease of Online booking'].value_counts())
print(airline_brute['Food and drink'].value_counts())
print(airline_brute['Online boarding'].value_counts())
print(airline_brute['Seat comfort'].value_counts())
print(airline_brute['Inflight entertainment'].value_counts())
print(airline_brute['On-board service'].value_counts())
print(airline_brute['Leg room service'].value_counts())
print(airline_brute['Baggage handling'].value_counts())
print(airline_brute['Checkin service'].value_counts())
print(airline_brute['Inflight service'].value_counts())
print(airline_brute['Cleanliness'].value_counts())
print(airline_brute['satisfaction'].value_counts())



#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário



#Salve o dataset atualizado se houver modificações.

airline.to_csv('airline_ajustado.csv', index=False)