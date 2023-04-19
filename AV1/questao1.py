#importe as bibliotecas necessárias
import pandas as pd

## Carregue o dataset definido para você
df = pd.read_csv(r'C:\Users\LAB1_00\Documents\Eduardo\IA\AV1\dataset\Hotel_Reservations.csv')


# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(df.isnull().sum())

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
novo_df = df.drop(["type_of_meal_plan", "arrival_year", "arrival_month", "arrival_date", "avg_price_per_room", "Booking_ID", "room_type_reserved", "market_segment_type"], axis = 1)
print(novo_df.isnull().sum())

#Print o dataframe final e mostre a distribuição de classes que você deve classificar
print(novo_df)
print(novo_df['booking_status'].value_counts())

#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário
mapa = {'Not_Canceled': 1, 'Canceled': 0}
novo_df['booking_status'] = novo_df['booking_status'].map(mapa)
#novo_df['booking_status'] = novo_df['booking_status'].astype(bool)
print(novo_df['booking_status'].value_counts())

#Salve o dataset atualizado se houver modificações.
novo_df.to_csv('Hotel_Reservations_ajustado.csv', index=False)