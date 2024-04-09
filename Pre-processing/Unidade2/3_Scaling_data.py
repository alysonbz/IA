from sk.learn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier


wine = load_wine_dataset()

# Inicialize o scale
scaler = StandardScaler

# Exclua do dataset a coluna Quality
X = wine.drop(['Quality'],axis=1)

# Normalize o dataset com scaler
X_norm = scaler.fit.transform(X)

# Obtenha as labels da coluna Quality
y = wine['Quality'].values

# Print a variância de X
print('variancia', X.var())

# Print a variânca do dataset X_norm
print('variancia do dataset normalizado', X_norm.var)

# Divida o dataset em treino e teste com amostragem estratificada
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, stratify=y, random_state=42)

# Inicialize o algoritmo KNN
knn = KNeighborsClassifier()

# Aplique a função fit do KNN
knn.fit(X_train, y_train)

# Verifique o acerto do classificador
print('score', knn.score(X_test, y_test))