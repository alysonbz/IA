import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from scipy.cluster.hierarchy import fcluster



# Carregando o dataset e análise (pré-processamento)
dataset = pd.read_csv(r"C:\Users\Aluno\Documents\Thais\IA\AV3\archive (1)\cancer_classification.csv")
print(dataset.describe())
dataset.describe()
dataset.info()
dataset.head()
dataset.isnull().sum()

# Primeiro vamos fazer a normalização
#mostrar a correlação entra a coluna principal e as demais colunas do dataset
dataset.corr()['benign_0__mal_1'].sort_values().plot(kind='bar')
X = dataset.drop(['benign_0__mal_1'], axis=1)
y = dataset['benign_0__mal_1'].values


normalizer = Normalizer()
kmeans = KMeans(n_clusters=2)
pipeline = make_pipeline(normalizer, kmeans)
pipeline.fit(X)
labels = pipeline.predict(X)
df = pd.DataFrame({'labels': labels, 'y': y})
print(df)

# Após a normalização, faremos o dendrograma
mergings = linkage(X, method='complete')
plt.figure(figsize=(12, 6))
dendrogram(mergings,
           labels=y,
           leaf_rotation=90,
           leaf_font_size=8)
plt.xlabel('Amostras')
plt.ylabel('Distância')
plt.title('Dendrograma')
plt.tight_layout()
plt.show()

#agora faremos o processo de clusterização

mergings = linkage(X, method='complete')
labels = fcluster(mergings, 200, criterion = 'distance')
print(labels)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from scipy.cluster.hierarchy import fcluster
import seaborn as sns
from sklearn.model_selection import train_test_split

# Carregando o dataset e análise (pré-processamento)
dataset = pd.read_csv(r"C:\Users\Aluno\Documents\Thais\IA\AV3\archive (1)\cancer_classification.csv")
dataset = dataset.sample(n = 60, replace = False)
print(dataset.describe())
dataset.describe()
dataset.info()
dataset.head()
dataset.isnull().sum()

# Primeiro vamos fazer a normalização
#mostrar a correlação entra a coluna principal e as demais colunas do dataset
dataset.corr()['benign_0__mal_1'].sort_values().plot(kind='bar')
X = dataset.drop(['benign_0__mal_1'], axis=1)
y = dataset['benign_0__mal_1'].values



normalizer = Normalizer()
n = normalizer.fit_transform(X)
df = pd.DataFrame(n, columns=X.columns)
df['benign_0__mal_1'] = y
print(df)

# Após a normalização, faremos o dendrograma
mergings = linkage(X, method='complete')
plt.figure(figsize=(10, 6))
dendrogram(mergings,
           labels=y,
           leaf_rotation=90,
           leaf_font_size=8)
plt.xlabel('Amostras')
plt.ylabel('Distância')
plt.title('Dendrograma')
plt.tight_layout()
plt.show()

#agora faremos o processo de clusterização

mergings = linkage(X, method='complete')
labels = fcluster(mergings, 2000, criterion = 'distance')
print(labels)
plt.show()

# já identificamos a coluna principal, agora vou ver  as diferentes categorias dentro dela
sns.countplot(data=df, x='benign_0__mal_1')