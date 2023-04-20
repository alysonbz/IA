
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.

df = pd.read_csv('flavors_of_cacao_ajustado.csv')

# Transforma a variável "Rating" em uma variável categórica
rating_bins = [0, 2.5, 3.5, 4.0, 5.0]
rating_labels = ['ruim', 'regular', 'bom', 'excelente']
df['Rating_cat'] = pd.cut(df['Rating'], bins=rating_bins, labels=rating_labels)

# Sem normalizar o conjunto de dados divida o dataset em treino e teste.
X = df.drop(['Rating', 'Rating_cat'], axis=1)
y = df['Rating_cat']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)


# Ajuste o modelo de KNN aos dados de treinamento.
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Avalie o modelo nos dados de teste.
accuracy = knn.score(X_test, y_test)
print('Acurácia do modelo KNN: {:.2f}'.format(accuracy))


