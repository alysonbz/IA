from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()
print(volunteer)

print("=== 1 - Mostre a dimensão do dataset volunteer ===")
print(volunteer.shape)

print("=== 2 - mostre os tipos de dados existentes no dataset ===")
#print(volunteer.info())

print("=== 3 - mostre quantos elementos do dataset estão faltando na coluna ===")
#print(volunteer.isnull().sum())

print("=== 4 - Exclua as colunas Latitude e Longitude de volunteer ===")
volunteer_cols = volunteer.drop(['Latitude', 'Longitude'], axis= 1)


print("=== 5 - Exclua as linhas com valores null da coluna category_desc de volunteer_cols")
volunteer_subset = volunteer_cols.dropna(subset=['category_desc'], axis = 0)
print(volunteer_subset)
print ('=== 6 - Print o shape do subset ===')
print(volunteer_subset.shape)