import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage, dendrogram

# carregar o conjunto de dados
df = pd.read_csv("Trojan_Detection.csv")

# Determinar o número de linhas
n_linhas = int(len(df) * 0.2)

# Reduzir o conjunto de dados
df_reduzido = df.sample(n=n_linhas)

# Salvar o novo conjunto de dados em um arquivo CSV
df_reduzido.to_csv('Trojan_30%.csv', index=False)

# Verificando se há valores nulos
print("\n verificação da existência de células vazias ou NaN")
print(df_reduzido.isna().sum())

# Transformando df_reduzido
map_df_reduzido = {
    'Trojan': 0,
    'Benign': 1
}
df_reduzido['Class'] = df_reduzido['Class'].map(map_df_reduzido)

df_atualizado = df_reduzido.drop(['Flow ID', ' Source IP', ' Destination IP', ' Timestamp'], axis=1)

# Excluindo valores NaN
df_atualizado = df_atualizado.dropna()

X = df_atualizado.drop(['Class'], axis=1)
y = df_atualizado["Class"].values
scaler = StandardScaler()
normalized = scaler.fit_transform(X)
mergings = linkage(normalized, method='complete')

# Plot the dendrogram
dendrogram(mergings,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()


# Clusterização - K_means
model = KMeans(n_clusters=3)
# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(df_atualizado)
# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'Class': y})
# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['y'])
# Display ct
print(ct)