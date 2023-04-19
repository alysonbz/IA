# Importe as bibliotecas necessárias.
from questao1 import avc_ajustado
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize, StandardScaler

# Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
avc_ajustado = avc_ajustado

# Normalize o conjunto de dados com normalização logarítmica e verifique a acurácia do knn.
X = avc_ajustado.drop('stroke', axis=1)
y = avc_ajustado['stroke']
X_normalizado_log = normalize(X, norm='l2')
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_normalizado_log, y, test_size=0.3, random_state=1, stratify=y)
knn = KNeighborsClassifier()
# Aplicando a função fit do knn
knn.fit(X_train_log, y_train_log)
y_pred_log = knn.predict(X_test_log)
acuracia_log = accuracy_score(y_test_log, y_pred_log)

# Normalize o conjunto de dados com normalização de média zero e variância unitária e verifique a acurácia do knn.
scaler = StandardScaler()
X_normalizado_scaler = scaler.fit_transform(X)
X_train_scaler, X_test_scaler, y_train_scaler, y_test_scaler = train_test_split(X_normalizado_scaler, y, test_size=0.3, random_state=1, stratify=y)

knn.fit(X_train_scaler, y_train_scaler)
y_pred_scaler = knn.predict(X_test_scaler)
acuracia_scaler = accuracy_score(y_test_scaler, y_pred_scaler)

# Print as duas acuracias lado a lado para comparar.
print('Acurácia com normalização logarítmica: {:.2f}%'.format(acuracia_log * 100))
print('Acurácia com normalização de média zero e variância unitária (scaler): {:.2f}%'.format(acuracia_scaler * 100))
