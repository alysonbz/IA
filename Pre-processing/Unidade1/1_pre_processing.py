from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print(volunteer.ndim)

#mostre os tipos de dados existentes no dataset
print(volunteer.info)

#mostre quantos elementos do dataset estão faltando na coluna locality
print(volunteer['locality'].isnull().sum())

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_cols = print(volunteer.drop(['Latitude', 'Longitude'], axis=1))

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
###volunteer_subset =

# Print o shape do subset



