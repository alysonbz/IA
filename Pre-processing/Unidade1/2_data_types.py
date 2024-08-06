from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()
print(volunteer)
print ('=== 1 - Print os primeiros elementos da coluna hits ===')
print(volunteer['hits'].head())

print('=== Print as caracteristicas da coluna hits ===')
print(volunteer.info(['hits']))

print('=== Converta a coluna hits para o tipo int ===')
volunteer['hits'] = volunteer['hits'].astype(int)

print('=== Print as caracteristicas da coluna hits novamente ===')
print(volunteer['hits'])