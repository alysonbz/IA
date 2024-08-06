# Importe as bibliotecas necessárias.
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# Carregue o dataset definido para você.
drug200_new = pd.read_csv('dataset/drug200_new.csv')
print(drug200_new.head())


# Normalize o conjunto de dados com normalização logarítmica  e verifique a acurácia do knn.
X = drug200_new.drop(columns=['Drug'], axis=1)
y = drug200_new['Drug']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

epsilon = 1e-9
X_train_safe = X_train + epsilon
X_test_safe = X_test + epsilon

log_transformer = FunctionTransformer(np.log1p, validate=True)
X_train_log = log_transformer.fit_transform(X_train_safe)
X_test_log = log_transformer.transform(X_test_safe)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_log, y_train)
y_pred_log = knn.predict(X_test_log)
accuracy_log = accuracy_score(y_test, y_pred_log)

# Normalize o conjunto de dados com normalização de media zero e variância unitária e verifique a acurácia do knn.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn.fit(X_train_scaled, y_train)
y_pred_scaled = knn.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)


# Print as duas acuracias lado a lado para comparar.
print(f'Acurácia com Normalização Logarítmica: {accuracy_log:.4f}')
print(f'Acurácia com Normalização de Média Zero e Variância Unitária: {accuracy_scaled:.4f}')