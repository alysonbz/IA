from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Print os primeiros elementos da coluna hits
print(volunteer["hits"])
print()

# Print as caracteristicas da coluna hits
print("Características da coluna hits:", volunteer["hits"].dtypes)
print()

# Converta a coluna hits para o tipo int
volunteer_int = volunteer["hits"].astype("int32")
print()

# Print as caracteristicas da coluna hits novamente
print("Coluna volunteer convertida para:", volunteer_int.dtypes)
print(volunteer_int.dtypes)