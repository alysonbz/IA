from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Exclua as colunas Latitude e Longitude de volunteer
print('\nExclui as colunas Latitude e Longitude de volunteer')
volunteer_new = volunteer.drop(['Latitude', 'Longitude'], axis = 1)
print(volunteer_new.info)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
print('\nExcluindo as linhas com valores null da coluna category_desc de volunteer_new')
volunteer = volunteer_new.dropna(subset=['category_desc'])
print(volunteer)

# mostre o balanceamento das classes em 'category_desc'
print('\nMostrando o balanceamento das classes em ´´category_desc´´ ')
print(volunteer['category_desc'].value_counts(),'\n','\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
print('\nCriando um DataFrame com todas as colunas, com exceção de ``category_desc``')
X = volunteer.drop('category_desc', axis=1)

# Crie um dataframe de labels com a coluna category_desc
print('\nCriando um dataframe de labels com a coluna category_desc')
y = volunteer[['category_desc']]

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
print('\nUtiliza a a amostragem stratificada para separar o dataset em treino e teste')
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
print(X_train, X_test, y_train, y_test)

# mostre o balanceamento das classes em 'category_desc' novamente
print('\nMostre o balanceamento das classes em ´´category_desc´´ novamente')
print(volunteer['category_desc'].value_counts(), '\n', '\n')