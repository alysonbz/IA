# Utilizando análise de variância do PCA. Reduza a dimensão para realizar uma classificação utilizando somente as colunas de maior variância.
# Aplique o mesmo método de classificação testado na questão 3.
# Gere os mesmos números que analisam o desempenho do classificador e verifique se houve melhoria no resultado.

# pacotes
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
import pandas as pd

# dados
df = pd.read_csv(r'C:\Users\Guilherme\Documents\G\IA\AV3\INCART 2-lead Arrhythmia Database.csv')
df = df.dropna()

# converter valores categóricos em valores numéricos
encoder = LabelEncoder()
df['type'] = encoder.fit_transform(df['type'])

# retirar amostra de 30% dos dados
sample_size = int(0.3 * len(df))
random_sample = df.sample(n=sample_size, random_state=42)

# selecionar os atributos relevantes
samples = random_sample.drop(['type', 'record'], axis=1)
types = random_sample['type'].values

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
plt.ylabel('variance')
plt.xticks(features)
plt.show()
