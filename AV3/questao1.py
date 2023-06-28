#Importando Bibliotecas

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage, dendrogram

# Carregar o conjunto de dados
data = pd.read_csv('csgo_round_snapshots.csv')

# Determinar o número de linhas que serão mantidas (50% das linhas originais)
num_linhas = int(len(data) * 0.3)

# Reduzir o conjunto de dados para o número de linhas desejado
csgo_round = data.sample(n=num_linhas)

# Salvar o novo conjunto de dados em um arquivo CSV
csgo_round.to_csv('csgo_round_snapshots_50percent.csv', index=False)

#Verificando se há valores nulos
print("\n Verificação da existência de células vazias ou NaN")
print(csgo_round.isna().sum())

#Transformando Round_winner
map_round_winner = {
    'T': 0,
    'CT': 1
}
csgo_round['round_winner'] = csgo_round['round_winner'].map(map_round_winner)

'''pd.set_option('display.max_columns', None)

print(csgo_round_sample)'''

#Salvando um novo dataset atualizado
csgo_round_df = csgo_round.drop(['map','bomb_planted'], axis = 1)
print(csgo_round_df)

#Excluindo valores NaN
csgo_round_df = csgo_round_df.dropna()

X = csgo_round_df.drop(['round_winner'], axis=1)
y = csgo_round_df["round_winner"].values

normalizando = normalize(X)
mergings = linkage(normalizando, method='complete')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels= y,
           leaf_rotation=90,
           leaf_font_size=6,
)
'''plt.show()'''

# Clusterização - K_means
model = KMeans(n_clusters=6)
labels = model.fit_predict(X)
df = pd.DataFrame({'labels': labels, 'round_winner': y})
ct = pd.crosstab(df['labels'], df['round_winner'])

# Display ct
print(ct)