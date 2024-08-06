# Importe as bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Carregue o dataset definido para você.
healthcare_data = pd.read_csv('dataset/healthcare-dataset-stroke-data-new.csv')
print(healthcare_data.head())

# Normalize o conjunto de dados com normalização logarítmica  e verifique a acurácia do knn.
X = healthcare_data.drop(columns=['stroke'], axis=1)
y = healthcare_data['stroke']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def log_normalize(X):
    return np.log1p(X)

log_transformer = FunctionTransformer(func=log_normalize, validate=False)
X_train_log = log_transformer.fit_transform(X_train)
X_test_log = log_transformer.transform(X_test)

# Treinar o modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_log, y_train)

# Fazer previsões e calcular acurácia
y_pred_log = knn.predict(X_test_log)
accuracy_log = accuracy_score(y_test, y_pred_log)
print(f'Acurácia com normalização logarítmica: {accuracy_log:.2f}')


# Normalize o conjunto de dados com normalização de media zero e variância unitária e verifique a acurácia do knn.
# Normalização de média zero e variância unitária
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinar o modelo KNN
knn.fit(X_train_scaled, y_train)

# Fazer previsões e calcular acurácia
y_pred_scaled = knn.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
print(f'Acurácia com normalização de média zero e variância unitária: {accuracy_scaled:.2f}')
print()
