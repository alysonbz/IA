import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score



#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
star_class_new = pd.read_csv(r"dataset\star_classification.csv")
#print(star_class_new.shape)
#Sem normalizar o conjunto de dados divida o dataset em treino e teste.
print(star_class_new['class'].value_counts(), "\n")
X = star_class_new.drop(["class"], axis=1)
y = star_class_new[["class"]].values

#y_train = star_class_new[["class"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.3)
print(y_train["class"].value_counts())


#Implemente o Knn exbindo sua acurácia nos dados de teste e mantenha sua parametrização default.
knn = KNeighborsRegressor()
knn.fit(X_train,y_train)
y_pred= knn.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print(accuracy)
print(knn.score(X_test, y_test))