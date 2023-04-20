
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
df = pd.read_csv('flavors_of_cacao_ajustado.csv')

# Transforme a variável de saída (Rating) em classes.
df['class'] = pd.cut(df['Rating'], bins=[0, 2.99, 3.49, 3.99, 4.49, 4.99, 5.49, 5.99], labels=[0, 1, 2, 3, 4, 5, 6])

# Remova a coluna Rating do dataset.
df.drop('Rating', axis=1, inplace=True)

# Normalize o conjunto de dados com normalização logarítmica e verifique a acurácia do knn.
numeric_cols = df.select_dtypes(include="number").columns

df[numeric_cols] = df[numeric_cols].applymap(lambda x: np.log(x) if x > 0 else x)

df = df.replace([np.nan, np.inf, -np.inf], np.nan).dropna()

X = df[numeric_cols].values
y = df['class'].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

score_log = knn.score(X_test, y_test)
print('Norm log:', score_log)

# Normalize o conjunto de dados com normalização de media zero e variância unitária e e verifique a acurácia do knn.
scaler = StandardScaler()

X_norm = pd.DataFrame(scaler.fit_transform(X), columns=numeric_cols)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)
print("Acurácia do modelo KNN:", accuracy)

# Print as duas acuracias lado a lado para comparar.

print("Acuracia da normalização logaritmica:", score_log)
print("Acurácia da normalização de media zero:", accuracy)




