from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(['latitude', 'longitude'], axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer_new.dropna()

# mostre o balanceamento das classes em 'category_desc'
#print(['category_desc'].__,'\n','\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
#X = volunteer.(, axis=1)

# Crie um dataframe de labels com a coluna category_desc
#y = [['']]

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
#X_train, X_test, y_train, y_test = __(__, __, stratify=__, random_state=42)

# mostre o balanceamento das classes em 'category_desc' novamente
