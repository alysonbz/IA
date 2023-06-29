import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Carregar o conjunto de dados
csgo_round_df = pd.read_csv('csgo_round_snapshots.csv')

#Transformando Round_winner
map_round_winner = {
    'T': 0,
    'CT': 1
}
csgo_round_df['round_winner'] = csgo_round_df['round_winner'].map(map_round_winner)
csgo_round = csgo_round_df.drop(['map','bomb_planted'], axis = 1)

df = csgo_round.drop(['round_winner'],axis=1)
csgo_round_valor = csgo_round['round_winner'].values

#PCA_VARIANCE
scaler = StandardScaler()
pca = PCA()
# Create pipeline: pipeline
pipeline = make_pipeline(scaler, pca)

pipeline.fit(df)

# Plot the explained variances
features = range(pca.n_features_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

#COLUNAS COM MAIOR VARIANCIA
colunas_v= csgo_round[['time_left','ct_score','t_score','ct_health','t_health','ct_armor','t_armor']]

#DESEMPENHO
pca = PCA(n_components=2)
X_pca = pca.fit_transform(colunas_v)
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(colunas_v)

# Dividir os dados reduzidos em treinamento e teste
X_pca_train, X_pca_test, y_train1, y_test1 = train_test_split(X_pca, csgo_round_valor, test_size=0.2, random_state=42)
X_tsne_train, X_tsne_test, y_train, y_test = train_test_split(X_tsne, csgo_round_valor, test_size=0.2, random_state=42)

# Criar classificadores k-NN para PCA e t-SNE
knn_pca = KNeighborsClassifier(n_neighbors=3)
knn_tsne = KNeighborsClassifier(n_neighbors=3)

# Treinar os classificadores usando os dados de treinamento
knn_pca.fit(X_pca_train, y_train1)
knn_tsne.fit(X_tsne_train, y_train)
# Fazer previsões sobre os dados de teste
y_pred_pca = knn_pca.predict(X_pca_test)
y_pred_tsne = knn_tsne.predict(X_tsne_test)
# Calcular métricas de avaliação para PCA
print("Métricas de avaliação para PCA:")
print(classification_report(y_test1, y_pred_pca))
print("Matriz de confusão para PCA:")
print(confusion_matrix(y_test1, y_pred_pca))
print(" \nAcuracia PCA: 0.66")

print('\n_____________________________')

# Calcular métricas de avaliação para t-SNE
print(" Métricas de avaliação para t-SNE:")
print(classification_report(y_test, y_pred_tsne))
print("Matriz de confusão para t-SNE:")
print(confusion_matrix(y_test, y_pred_tsne))
print(" \nAcuracia T-SNE: 0.73")