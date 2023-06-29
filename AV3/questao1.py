import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

oleo_df= pd.read_csv(r"C:\Users\LAB1_00\Desktop\SAVIO\IA\AV3\oil_spill.csv")

#normalização dos dados
scaler = StandardScaler()
df_normalized = scaler.fit_transform(oleo_df)

movements = oleo_df.drop(['target'],axis=1)
companies = oleo_df['target'].values

# Normalize the movements: normalized_movements
normalized_movements = normalize(movements)

# Calculate the linkage: mergings
mergings = linkage(normalized_movements, method='complete')

# Plot the dendrogram
dendrogram(mergings, labels=companies,leaf_rotation=90, leaf_font_size=6)
plt.show()

# Clusterização - K_means
model = KMeans(n_clusters=3)
# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(movements)
# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'class_values': companies})
# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['class_values'])
# Display ct
print(ct)


#exportando
oleo_df.to_csv('oil_spill.csv')

