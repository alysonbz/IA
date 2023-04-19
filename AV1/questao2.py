# Importe as bibliotecas necessárias.
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
gender_one = pd.read_csv('gender_final.csv')
print("\n Dataset: Gender Atualizado")
print(gender_one)

# Sem normalizar o conjunto de dados divida o dataset em treino e teste.
X = gender_one[['long_hair','forehead_height_cm','nose_wide', 'nose_long']].values
y = gender_one['gender'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Implemente o Knn exibindo sua acurácia nos dados de teste e mantenha sua parametrização default.
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print("Resultado acuracia:")
print(knn.score(X_test, y_test))