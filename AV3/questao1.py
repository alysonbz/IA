# 1 - Faça uma análise do dataset utilizando dendograma.
# 2 - Verifique as possibilidades de clusterização e aplique o k-medias.
# Observe os resultados e descreva sua iterpretaçãono relatório.
# Dica: Observe se há necessidade de normalização dos dados nas colunas ou nas linhas.

# pacotes
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans


# dados
df = pd.read_csv(r'C:\Users\guilh\OneDrive\Documentos\GD\INCART 2-lead Arrhythmia Database.csv')
df = df.dropna()

# conhecer dados
print(df.columns)
print(df.head())
print(df.info())

# retirar amostra de 30% dos dados
sample_size = int(0.3 * len(df))
random_sample = df.sample(n=sample_size, random_state=42)

# selecionar os atributos relevantes
samples = random_sample.drop(['type', 'record'], axis=1).select_dtypes(exclude="object")
types = random_sample['type'].values

# normalizar os dados
normalized = normalize(samples)

#1
# calcular o linkage utilizando os dados normalizados
mergings = linkage(normalized, method='complete')

# plotar dendrogram
dendrogram(mergings,
           labels=types,
           leaf_rotation=45,
           leaf_font_size=6,
)
plt.show()

# 2
# verificar o k adequado para o k-means utilizando a inertia
ks = range(1, 6)
inertias = []
for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(normalized)
    inertias.append(model.inertia_)

# plotar gráfico da inertia
plt.plot(ks, inertias, '-o')
plt.xlabel('numero de clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

# o cotovelo do gráfico indicou k = 2

# instanciar k-means com 2 clusters
model = KMeans(n_clusters=2)

# ajustar o modelo e obter as labels do cluster
labels = model.fit_predict(normalized)
print(labels)

# analisar se a clusterização ocoreru adequadamente utilizando o crosstable
dfct = pd.DataFrame({'labels': labels, 'types': types})
ct = pd.crosstab(dfct['labels'], dfct['types'])
print(ct)