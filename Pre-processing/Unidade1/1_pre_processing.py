from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print(volunteer.ndim)
print('\n')

# mostre os tipos de dados existentes no dataset
print(volunteer.dtypes)
print('\n')

# mostre quantos elementos do dataset estão faltando na coluna
print(volunteer.isna().sum())
print('\n')

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_cols = volunteer.drop(["Latitude", "Longitude"], axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
volunteer_cols.dropna(subset=["category_desc"], inplace=True)

# Print o shape do subset
print(volunteer.shape)
