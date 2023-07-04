import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import normalize

df = pd.read_csv('smoke_detection_iot.csv')

df1 = df.sample(frac=0.8, random_state=42)  # Seleciona 80% dos dados aleatoriamente

df1 = df1.dropna()

X = df1.drop(['FireAlarm'], axis=1)
y = df1['FireAlarm'].values

normalized = normalize(X)

# Calculate the linkage: mergings
mergings = linkage(normalized, method='complete')

# Plot the dendrogram
dendrogram(mergings,
           labels=y,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()

# Clusterização - K_means
model = KMeans(n_clusters=3)
# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(X)
# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'FireAlarm': y})
# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['FireAlarm'])
# Display ct
print(ct)