import pandas as pd
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

glass_df = pd.read_csv("glass.csv")

#Visualizar informações do dataset
print(glass_df.info())
#Verificar a existencia de valores nulos.
print(glass_df.isnull().sum())
# Criar uma instância do LabelEncoder
label_encoder = LabelEncoder()

# Percorrer as colunas do dataset
for coluna in glass_df.columns:
    # Verificar se a coluna contém valores de string
    if glass_df[coluna].dtype == 'object':
       # Aplicar o LabelEncoder na coluna
        glass_df[coluna] = label_encoder.fit_transform(glass_df[coluna])


# DENDOGRAMA
cogu = glass_df.drop(['Type'], axis=1)
clas = glass_df['Type'].values

# Normalizar os dados
normalized_test = normalize(cogu)

# Calcular o linkage: mergings
mergings = linkage(normalized_test, method='complete')

# Plotar o dendrograma
plt.figure(figsize=(10, 6))
dendrogram(mergings, labels=clas, leaf_rotation=90, leaf_font_size=8)

plt.title('Dendrograma')
plt.xlabel('Amostras')
plt.ylabel('Distância')
plt.show()

# Clusterização e aplique o k-means
model = KMeans(n_clusters=3)

# Use fit_predict em model
labels = model.fit_predict(cogu)

# Crie um DataFrame com as labels e diagnosis como colunas
df = pd.DataFrame({'labels': labels, 'Type': clas})

# Crie uma tabela de contingência (crosstab): ct
ct = pd.crosstab(df['labels'], df['Type'])

# Exiba a tabela de contingência
print(ct)