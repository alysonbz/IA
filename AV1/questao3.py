#Importe as bibliotecas necessárias.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize, StandardScaler

#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
from questao1 import hda

#Normalize o conjunto de dados com normalização logarítmica e verifique a acurácia do knn.
X = hda.drop('HeartDisease', axis=1)
y = hda['HeartDisease']
X_norm_log = normalize(X, norm='l2')
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_norm_log, y, test_size=0.3, random_state=42, stratify=y)

knn = KNeighborsClassifier()
knn.fit(X_train_log, y_train_log)

y_predlog = knn.predict(X_test_log)
acuracialog = accuracy_score(y_test_log, y_predlog)

#Normalize o conjunto de dados com normalização de media zero e variância unitária e e verifique a acurácia do knn.
scaler = StandardScaler()

X_norm_scaler = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_norm_scaler, y, test_size=0.3, random_state=42, stratify=y)
knn.fit(X_train, y_train)
y_pred_scaler = knn.predict(X_test)
acc_scaler = accuracy_score(y_test, y_pred_scaler)

#Print as duas acuracias lado a lado para comparar.
print('Acurácia com normalização de log: {:.2f}%'.format(acuracialog * 100))
print('Acurácia com normalização de média zero e variância unitária: {:.2f}%'.format(acc_scaler * 100))


