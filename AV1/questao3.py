# Importe as bibliotecas necessárias.
from questao1 import customer_ajustado
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize, StandardScaler


# Normalize o conjunto de dados com normalização logarítmica e verifique a acurácia do knn.
X = customer_ajustado.drop('label', axis=1)
y = customer_ajustado['label']
X_norm_log = normalize(X, norm='l2')
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_norm_log, y, test_size=0.3, random_state=42, stratify=y)
knn = KNeighborsClassifier()
# Aplicando a função fit do knn
knn.fit(X_train_log, y_train_log)
y_pred_log = knn.predict(X_test_log)
acuracia_log = accuracy_score(y_test_log, y_pred_log)

# Normalize o conjunto de dados com normalização de média zero e variância unitária e verifique a acurácia do knn.
scaler = StandardScaler()
X_norm_scaler = scaler.fit_transform(X)
X_train_scaler, X_test_scaler, y_train_scaler, y_test_scaler = train_test_split(X_norm_scaler, y, test_size=0.3, random_state=42, stratify=y)

knn.fit(X_train_scaler, y_train_scaler)
y_pred_scaler = knn.predict(X_test_scaler)
acc_scaler = accuracy_score(y_test_scaler, y_pred_scaler)

# Print as duas acuracias lado a lado para comparar.
print('Acurácia com normalização de log: {:.2f}%'.format(acuracia_log * 100))
print('Acurácia com normalização de média zero e variância unitária: {:.2f}%'.format(acc_scaler * 100))