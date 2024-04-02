from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print("Dimensão do dataset volunteer:", volunteer.shape)

# Mostre os tipos de dados existentes no dataset
print("Tipos de dados existentes no dataset:", volunteer.info)

# Mostre quantos elementos do dataset estão faltando na coluna locality
print(volunteer['locality'].isnull().sum())

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_cols = volunteer.drop(['Latitude', 'Longitude'], axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
volunteer_subset = volunteer_cols.dropna(subset=['category_desc'])

# Print o shape do subset
print("Shape do subset após exclusão das linhas nulas: ", volunteer_subset.shape)


