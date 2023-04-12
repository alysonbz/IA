import pandas as pd

from src.utils import  load_df1_unidade2,load_df2_unidade2


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = pd.read_csv('iris.data.csv')
print(iris.value_counts())



# divida o dataset em treino e teste
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
knn = KNeighborsClassifier()

# Aplique a função fit do knn
#knn.fit(X_train, y_train)

# mostre o acerto do algoritmo
#print(knn.score(X_test,y_test))