import pandas as pd
from sklearn.preprocessing import  Normalizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

########## LINK DO COLAB: https://colab.research.google.com/drive/1R96LXaCIBnbV_Of4jVFrtNbATqD3KB18?usp=sharing
#lendo o dataset
estado_do_olho0 = pd.read_csv(r"C:\Users\LAB1_00\Desktop\BIANCA\ia_novo\IA\AV3\archive (2)\EEG_Eye_State_Classification.csv")
estado_do_olho = estado_do_olho0.sample(n=1498, replace=False)  # Com 10% do dataset

# Separando os atributos das classes
X = estado_do_olho.drop('eyeDetection', axis=1)
y = estado_do_olho['eyeDetection']

# Aplicando a normalização
scaler = Normalizer()
normal = scaler.fit_transform(X)

# Criando um novo DataFrame com os dados normalizados
estado_do_olho2 = pd.DataFrame(normal, columns=X.columns)

# k-means
k = 2
kmeans = KMeans(n_clusters=k)
kmeans.fit(estado_do_olho2)

labels = kmeans.labels_

# Centróides dos clusters
centroids = kmeans.cluster_centers_

# Plotando os resultados
plt.figure(figsize=(8, 6))
plt.scatter(estado_do_olho2.iloc[:, 0], estado_do_olho2.iloc[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='red', s=100, label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('label X')
plt.ylabel('label Y')
plt.legend()
plt.show()

print("Etiquetas de cluster:")
print(labels)

print("\nCentróides:")
print(centroids)