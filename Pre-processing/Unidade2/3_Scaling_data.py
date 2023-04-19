# Import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier

print('Dataset wine')
wine = load_wine_dataset()

# Inicialize o scale
print('\nInicialize o scale')
scaler = StandardScaler()

# exclua do dataset a coluna Quality
print('\nExcluir do dataset a coluna Quality')
X = wine.drop(['Quality'],axis=1)

#normalize o dataset com scaler
print('\nNormalizando o dataset com scaler')
X_norm = scaler.fit_transform(X)

#obtenha as labels da coluna Quality
print('\nObter as labels da coluna Quality')
y = wine['Quality'].values

#print a variância de X
print('\nVariância: ', X.var())

#print a variânca do dataset X_norm
print('\nVariância do dataset normalizado', X_norm.var())

# Divida o dataset em treino e teste com amostragem estratificada
print('\nDividindo o Dataset em treino e teste com amostragem estratificada')
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

#inicialize o algoritmo KNN
print('\nInicializando o algoritmo KNN')
knn = KNeighborsClassifier()

# Aplique a função fit do KNN
print('\nAplique a função fit do KNN')
knn.fit(X_train, y_train)

# Verifique o acerto do classificador
print('Verificando o acerto do classificador')
print('Score: ', knn.score(X_test, y_test))