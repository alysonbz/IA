import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import normalize

df = pd.read_csv('smoke_detection_iot.csv')

df1 = df.sample(frac=0.4, random_state=42)  # Seleciona 20% dos dados aleatoriamente

df1 = df1.dropna()
print(df1)

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
