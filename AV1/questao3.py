import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from src.utils import load_new_customer_dataset

# Carregue o dataset definido para você.
new_customer = load_new_customer_dataset()

# Separar features e target
X = new_customer.drop(columns=['label'], axis=1)
y = new_customer['label']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Adicionar um valor pequeno para evitar log de zero ou valores negativos
epsilon = 1e-9
X_train_safe = X_train + epsilon
X_test_safe = X_test + epsilon

# Normalização Logarítmica
log_transformer = FunctionTransformer(np.log1p, validate=True)
X_train_log = log_transformer.fit_transform(X_train_safe)
X_test_log = log_transformer.transform(X_test_safe)

# Treinamento e teste do KNN com normalização logarítmica
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_log, y_train)
y_pred_log = knn.predict(X_test_log)
accuracy_log = accuracy_score(y_test, y_pred_log)

# Normalização de Média Zero e Variância Unitária
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinamento e teste do KNN com normalização de média zero e variância unitária
knn.fit(X_train_scaled, y_train)
y_pred_scaled = knn.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)

# Printar as acurácias
print(f'Acurácia com Normalização Logarítmica: {accuracy_log:.4f}')
print(f'Acurácia com Normalização de Média Zero e Variância Unitária: {accuracy_scaled:.4f}')
