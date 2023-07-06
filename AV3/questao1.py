
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

data= pd.read_csv("C:/Users/Luciana/OneDrive/Documentos/IA PROJECTS/ap3/clinvar_conflicting.csv",low_memory=False)
print(data)

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(data.isna().sum(), '\n')

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
#==============================================================================================================================================================================
data = data.drop(['CLNDISDBINCL', 'CLNDNINCL', 'CLNSIGINCL', 'CLNVI', 'SSR', 'INTRON', 'EXON', 'SYMBOL', 'Feature_type', 'Feature', 'BIOTYPE',
                  'cDNA_position', 'CDS_position', 'Protein_position', 'Amino_acids', 'Codons', 'DISTANCE', 'STRAND', 'BAM_EDIT', 'SIFT', 'PolyPhen',
                  'MOTIF_NAME', 'HIGH_INF_POS', 'MOTIF_SCORE_CHANGE', 'LoFtool', 'CADD_PHRED', 'CADD_RAW', 'BLOSUM62', 'MC', 'MOTIF_POS',
                  'CLNDISDB', 'CLNDN', 'CLNHGVS', 'REF', 'ALT', 'Allele', 'Consequence', 'CHROM'], axis=1)
#'==============================================================================================================================================================================
print("Frequência de cada valor presente na coluna:\n", data['AF_ESP'].value_counts(),'\n','\n')
data['AF_ESP'] = data['AF_ESP'].astype(int)

print("Frequência de cada valor presente na coluna:\n", data['AF_EXAC'].value_counts(),'\n','\n')
data['AF_EXAC'] = data['AF_EXAC'].astype(int)

print("Frequência de cada valor presente na coluna:\n", data['AF_TGP'].value_counts(),'\n','\n')
data['AF_TGP'] = data['AF_TGP'].astype(int)

print("Frequência de cada valor presente na coluna:\n", data['CLNVC'].value_counts(),'\n','\n')
data['CLNVC'] = data['CLNVC'].replace({'single_nucleotide_variant': 0, 'Deletion':1, 'Duplication':2, 'Inversion':3, 'Indel':4, 'Insertion':5, 'Microsatellite':6})

print("Frequência de cada valor presente na coluna:\n", data['IMPACT'].value_counts(),'\n','\n')
data['IMPACT'] = data['IMPACT'].replace({'MODERATE': 0, 'LOW':1, 'MODIFIER':2, 'HIGH':3})
#==============================================================================================================================================================================
#Salve o dataset atualizado se houver modificações.
database=data
print("Data Frame atualizado:\n",database)

# 10% do dataset
df = database.sample(frac=0.1, random_state=42)

df = df.dropna()

X = df.drop(['CLASS'], axis=1)
y = df['CLASS'].values

normalized = normalize(X)

# Calcule a ligação: mergings
mergings = linkage(normalized, method='complete')

# Plotar o dendrogram
dendrogram(mergings,
           labels=y,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.figure(figsize=(10, 6))
dendrogram(mergings)
plt.title('Dendrograma')
plt.xlabel('Amostras')
plt.ylabel('Distância')
plt.show()

# Clusterização - K_means
model = KMeans(n_clusters=3)

# Use fit_predict para ajustar o modelo e obter rótulos de labels: labels
labels = model.fit_predict(X)

# Crie um DataFrame com rótulos e variedades como colunas: df
df = pd.DataFrame({'labels': labels, 'CLASS': y})

# Criar crosstab: ct
ct = pd.crosstab(df['labels'], df['CLASS'])

# Exibir ct
print(ct)




