# Importe as bibliotecas necessárias.
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
df = pd.read_csv('Hotel_Reservations_ajustado.csv')

# Sem normalizar o conjunto de dados divida o dataset em treino e teste.

X = df.drop('booking_status', axis=1)

y = df['booking_status'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Implemente o Knn exbindo sua acurácia nos dados de teste e mantenha sua parametrização default.

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

print("A acuracia de teste knn foi de: ",knn.score(X_test, y_test))