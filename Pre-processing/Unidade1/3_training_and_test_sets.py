from sklearn.model_selection import train_test_split

from src.utils import load_volunteer_dataset


volunteer = load_volunteer_dataset()

# Exclua as colunas Latitude e Longitude de volunteer
print("\n--- TODAS AS COLUNAS ---\n", volunteer.columns)

volunteer_new = volunteer.drop(["Latitude", "Longitude"], axis=1)

print("\n--- APÓS REMOVER AS COLUNAS ---\n", volunteer_new.columns)
print()


# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer_new = volunteer_new.dropna(subset=["category_desc"])


# mostre o balanceamento das classes em 'category_desc'
print(volunteer_new['category_desc'].value_counts(), '\n','\n')


# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer_new.drop(["category_desc"], axis=1)
print("\n--- NOVO DATAFRAME SEM 'category_desc' ---")
print(X.columns)


# Crie um dataframe de labels com a coluna category_desc
y = volunteer_new[['category_desc']]
print("\n--- DATAFRAME DE LABELS ---")
print(y.head())
print()

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# mostre o balanceamento das classes em 'category_desc' novamente
print(y_train['category_desc'].value_counts())
print()
print(y_test['category_desc'].value_counts())
