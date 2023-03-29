from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

print("\nPrint dos primeiros elementos da coluna hits:")
print(volunteer['hits'].head())

print("\nPrint das caracteristicas da coluna hits:")
print(volunteer['hits'].info())

print("\nConverte a coluna hits para o tipo int:")
print(volunteer['hits'] == volunteer['hits'].astype("int64"))

print("\nPrint das caracteristicas da coluna hits novamente:")
print(volunteer['hits'].info())