# 1 - Utilizando análise de variância do PCA. Reduza a dimensão para realizar uma classificação utilizando somente as colunas de maior variância.
# 2 - Aplique o mesmo método de classificação testado na questão 3.
# Gere os mesmos números que analisam o desempenho do classificador e verifique se houve melhoria no resultado.

# pacotes
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.pipeline import make_pipeline

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

# 1
# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, pca)

# Fit the pipeline to 'samples'
pipeline.fit(samples)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variância')
plt.xticks(features)
plt.show()