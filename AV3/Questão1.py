import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# Carregar o dataset
df1= pd.read_csv(r"mitbih_train.csv")

print(df1.head())
print(df1.info())
# Selecionar as colunas relevantes para a normalização

# Selecionar as colunas relevantes para a normalização
columns = ['1.000000000000000000e+00',
            '9.003241658210754395e-01',
            '3.585899472236633301e-01',
            '5.145867168903350830e-02',
          '4.659643396735191345e-02']  # substituir pelas colunas relevantes do dataset

# Criar um dataframe apenas com as colunas selecionadas
df1_selected = df1[columns]

# Normalizar as colunas
scaler = StandardScaler()
df_normalized = scaler.fit_transform(df1_selected)
# Normalizar as linhas
df1_normalized = df1_selected.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
# Criar uma matriz de distâncias
dist_matrix = linkage(df1_normalized, method='ward')

# Plotar o dendrograma
plt.figure(figsize=(10, 6))
dendrogram(dist_matrix)
plt.title('Dendrograma')
plt.xlabel('Índice das amostras')
plt.ylabel('Distância')
plt.show()
