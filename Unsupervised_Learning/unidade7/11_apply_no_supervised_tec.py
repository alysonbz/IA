from src.utils import load_fish_dataset
from sklearn.decomposition import PCA
from src.utils import load_fish_dataset
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold


fish= load_fish_dataset()
samples = fish.drop(['specie'],axis=1)
pca = PCA(n_components=2)


# Ajustar a instância do PCA às amostras dimensionadas
pca.fit(samples)


# Transforme as amostras dimensionadas: características do pca
tramsformer = pca.transform(samples)


# Imprima a forma de pca_features
print(tramsformer.shape)

#Matriz de confusão
X = tramsformer
y = fish['specie']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)
knn = KNeighborsClassifier(n_neighbors=6)


# Ajustar o modelo aos dados de treinamento
knn.fit(X_train, y_train)


# Preveja os rótulos dos dados de teste: y_pred
y_pred = knn.predict(X_test)


# Gera a matriz de confusão e o relatório de classificação
print(confusion_matrix(y_test, y_pred))

#Relatório de classificação
print(classification_report(y_test, y_pred))


#Acuracia
kf = KFold(n_splits=6, shuffle=True, random_state=5)
knn = KNeighborsClassifier()
cv_scores = cross_val_score(knn, X, y, cv=kf)
print(cv_scores)

