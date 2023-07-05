# Utilizando os dados da questão 2, aplique algum método de classificação e gere números que quantificam o desempenho deste.
# Compare os números classificando o dataset reduzido pelo PCA e pelo T-SNE.

# pacotes
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# dados
df = pd.read_csv(r'C:\Users\guilh\OneDrive\Documentos\GD\INCART 2-lead Arrhythmia Database.csv')
df = df.dropna()

encoder = LabelEncoder()
df['type'] = encoder.fit_transform(df['type'])

sample_size = int(0.3 * len(df))
random_sample = df.sample(n=sample_size, random_state=42)

samples = random_sample.drop(['type', 'record'], axis=1).select_dtypes(exclude="object")
types = random_sample['type'].values

normalized = normalize(samples)


# TSNE
# Aplicar o t-SNE nos dados
tmodel = TSNE(n_components=2)

# aplicar o fit_transform nos dados normalizados: tsne_features
tsne_features = tmodel.fit_transform(normalized)

# Dividir os dados em treinamento e teste
Xt_train, Xt_test, yt_train, yt_test = train_test_split(tsne_features, types, test_size=0.2, random_state=42)

# Instanciar o modelo de regressão logística
model = LogisticRegression()

# Treinar o modelo com os dados de treinamento
model.fit(Xt_train, yt_train)

# Avaliar a acurácia do modelo nos dados de teste
accuracy = model.score(Xt_test, yt_test)
print("Acurácia t-SNE - Regressão Logística: {:.2f}%".format(accuracy * 100))


# PCA
# Aplicar o PCA para reduzir a dimensionalidade para 2 componentes
pmodel = PCA(n_components=2)
pca_features = pmodel.fit_transform(normalized)

# Dividir os dados em treinamento e teste
Xp_train, Xp_test, yp_train, yp_test = train_test_split(pca_features, types, test_size=0.2, random_state=42)

# Treinar o modelo com os dados de treinamento
model.fit(Xp_train, yp_train)

# Avaliar a acurácia do modelo nos dados de teste
accuracy = model.score(Xp_test, yp_test)
print("Acurácia PCA - Regressão Logística: {:.2f}%".format(accuracy * 100))
