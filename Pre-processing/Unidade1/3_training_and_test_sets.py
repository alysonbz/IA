from sklearn.model_selection import train_test_split

from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

print('Exclua as colunas Latitude e Longitude de volunteer')
volunteer_new = volunteer.drop(columns=['Latitude','Longitude'])

print('Exclua as linhas com valores null da coluna category_desc de volunteer_new')
volunteer = volunteer_new.dropna(subset=['category_desc'])

print('=== mostre o balanceamento das classes em category_desc ===')
print(volunteer['category_desc'].value_counts(),'\n','\n')

print('Crie um DataFrame com todas as colunas, com exceção de ``category_desc``')
X = volunteer.drop(['category_desc'], axis=1)
print(X)

print('Crie um dataframe de labels com a coluna category_desc')
y =  volunteer[['category_desc']]

print('Utiliza a a amostragem stratificada para separar o dataset em treino e teste')
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

print('mostre o balanceamento das classes em category_desc novamente')
print(y_train['category_desc'].value_counts())