import pandas as pd
from src.utils import load_volunteer_dataset

volunteer = pd.DataFrame('dataset/archive/opportunities.csv')

# Mostre a dimensão do dataset volunteer

print("Tamanho do dataset", volunteer.shape)

#mostre os tipos de dados existentes no dataset

print("Os tipos de dados do dataset", volunteer.info)

#mostre quantos elementos do dataset estão faltando na coluna

print(volunteer['locality'].isnull().sum())

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_cols = volunteer.drop(['Latitude', 'Longitude'], axis=1)


# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
'''volunteer_subset ='''

# Print o shape do subset


