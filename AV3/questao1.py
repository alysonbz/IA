import pandas as pd
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

cancer_df = pd.read_csv("breast-cancer.csv")

#Visualizar informações do dataset
print(cancer_df.info())

# Criar uma instância do LabelEncoder
label_encoder = LabelEncoder()

# Percorrer as colunas do dataset
for coluna in cancer_df.columns:
    # Verificar se a coluna contém valores de string
    if cancer_df[coluna].dtype == 'object':
        # Aplicar o LabelEncoder na coluna
        cancer_df[coluna] = label_encoder.fit_transform(cancer_df[coluna])


# DENDOGRAMA
cogu = cancer_df.drop(['diagnosis'], axis=1)
clas = cancer_df['diagnosis'].values

# Normalizar os dados
normalized_test = normalize(cogu)

# Calcular o linkage: mergings
mergings = linkage(normalized_test, method='complete')

# Plotar o dendrograma
plt.figure(figsize=(10, 6))
dendrogram(mergings,
           labels=clas,
           leaf_rotation=90,
           leaf_font_size=8)

plt.title('Dendrograma')
plt.xlabel('Amostras')
plt.ylabel('Distância')
plt.show()

# Clusterização e aplique o k-means
model = KMeans(n_clusters=3)

# Use fit_predict em model
labels = model.fit_predict(cogu)

# Crie um DataFrame com as labels e diagnosis como colunas
df = pd.DataFrame({'labels': labels, 'diagnosis': clas})

# Crie uma tabela de contingência (crosstab): ct
ct = pd.crosstab(df['labels'], df['diagnosis'])

# Exiba a tabela de contingência
print(ct)

