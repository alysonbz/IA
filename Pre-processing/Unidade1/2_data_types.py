from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Print os primeiros elementos da coluna hits
print(volunteer["hits"].head())
print()

# Print as caracteristicas da coluna hits
print("\nCaracterísticas da coluna hits:", volunteer["hits"].dtypes)

# Converta a coluna hits para o tipo int
print("\nConvertendo a coluna hits para int..")
volunteer['hits'] = volunteer["hits"].astype(int)

# Print as caracteristicas da coluna hits novamente
print("\nColuna volunteer convertida para:", volunteer['hits'].dtypes)
