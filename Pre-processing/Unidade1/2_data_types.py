from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Print os primeiros elementos da coluna hits
print(volunteer["hits"].head())

# Print as caracteristicas da coluna hits
print(volunteer["hits"].info())

# Converta a coluna hits para o tipo int
# Ela ja estava como int, então para os fins da questão mudei para float
volunteer["hits"] = volunteer["hits"].astype(float)

# Print as caracteristicas da coluna hits novamente
print(volunteer["hits"].info())
