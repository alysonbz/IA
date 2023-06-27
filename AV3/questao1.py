import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.cluster.hierarchy import dendrogram
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

########## pré processamento
#lendo o dataset
estado_do_olho = pd.read_csv(r"C:\Users\LAB1_00\Desktop\BIANCA\ia_novo\IA\AV3\archive (2)\EEG_Eye_State_Classification.csv")
print(estado_do_olho)

#printando as classes
print(estado_do_olho["eyeDetection"].value_counts())

#'1'indica o estado de olho fechado e '0'o estado de olho aberto.

#################### KNN pq é classificação e depois comparar
# Dividir os dados em atributos (X) e rótulos (y)
X = estado_do_olho.drop("eyeDetection", axis=1)
y = estado_do_olho["eyeDetection"]

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo k-NN
knn = KNeighborsClassifier(n_neighbors=5)

# Treinar o modelo
knn.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = knn.predict(X_test)

# Calcular a acurácia das previsões
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do k-NN:", accuracy)




#dendrogram(mergings,
         #  labels= ,
        #   leaf_rotation=90,
       #    leaf_font_size=8,
#)
#plt.show()
