# Importe as bibliotecas necessárias.
from questao1 import avc_ajustado
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
avc_ajustado = avc_ajustado

# Sem normalizar o conjunto de dados divida o dataset em treino e teste.
X = avc_ajustado.drop('stroke', axis=1)
y = avc_ajustado['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Implemente o Knn exbindo sua acurácia nos dados de teste e mantenha sua parametrização default.
knn = KNeighborsClassifier()
# Aplicando a função fit do knn
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)
# Mostrando o acerto do algoritmo
print('\nAcurácia antes da normalização:', knn.score(X_test, y_test))
print('ou {:.2f}%'.format(acuracia * 100))