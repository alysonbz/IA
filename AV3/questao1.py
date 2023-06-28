#importe as bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans

#OBS
"""No conjunto de dados construído para este domínio, o atributo histórico familiar tem valor 1 se alguma dessas doenças foi observada na família e 0 caso contrário. O recurso de idade simplesmente representa a idade do paciente.
Todas as outras características clínicas e histopatológicas receberam um grau na faixa de 0 a 3. Aqui, 0 indica que a característica não estava presente, 3 indica a maior quantidade possível e 1, 2 indicam os valores intermediários relativos."""

## Carregue o dataset definido para você
derm = pd.read_csv('dermatology_database_1.csv')
print('\n DataSet: Conjunto de dados de dermatologia')
print(derm)

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
derm['age'] = derm['age'].replace('?', np.nan)
"print(derm.isna().sum())"
derm1 = derm.dropna(subset=['age'])


# DENDOGRAMA
derma = derm1.drop(['class'], axis=1)
class_values = derm1['class'].values
# Normalizar os dados
normalized_data = normalize(derma)
# Calculate the linkage: mergings
mergings = linkage(normalized_data, method='complete')
# Plotar o dendrograma
plt.figure(figsize=(10, 6))
dendrogram(mergings, labels=class_values, leaf_rotation=90, leaf_font_size=8)
plt.title('Dendrograma')
plt.xlabel('Amostras')
plt.ylabel('Distância')
plt.show()


# Clusterização - K_means
model = KMeans(n_clusters=3)
# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(derma)
# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'class_values': class_values})
# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['class_values'])
# Display ct
print(ct)






"""#Clusterização
# Use fcluster to extract labels: labels
labels = fcluster(mergings, 1.2, criterion='distance')
# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels':labels, 'class_values': class_values})
# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['class_values'])
# Display ct
print(ct)"""












norma_novo = derm1
norma_novo.to_csv('norma_novo.csv')


