from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
df = pd.read_csv('Hotel_Reservations_ajustado.csv')

# Normalize o conjunto de dados com normalização logarítmica e verifique a acurácia do knn.
numeric_cols = df.select_dtypes(include="number").columns.drop("booking_status")

df[numeric_cols] = df[numeric_cols].applymap(lambda x: np.log(x) if x > 0 else x)

df = df.replace([np.nan, np.inf, -np.inf], np.nan).dropna()
X = df[numeric_cols].values
y = df['booking_status'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# verificar se algum valor na matriz X_log é infinito

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

print(knn.score(X_test, y_test))

# Normalize o conjunto de dados com normalização de media zero e variância unitária e e verifique a acurácia do knn.
scaler = StandardScaler()

X_norm= pd.DataFrame(scaler.fit_transform(X), columns=numeric_cols)
print(X.var())
print(X_norm.var())


# Print as duas acuracias lado a lado para comparar.