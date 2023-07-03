import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans



smoking = pd.read_csv("smoking.csv")

map_gender = {
    'F': 0,
    'M': 1
}
smoking['gender'] = smoking['gender'].map(map_gender)

pd.set_option('display.max_columns', None)
print(smoking)

# Verificando se existe valores vazios ou Nan
print("\nVerificando se existe células vazias ou Nan")
print(smoking.isna().sum())

smoking_df = smoking.drop(['tartar','oral'], axis=1)
smoking_df.to_csv('smoking_df.csv')
print(smoking_df)

def load_smoking_data():
    X = smoking_df.drop(['smoking'], axis=1)
    y = smoking_df['smoking'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

X_train, df, y_train, smoking_valor = load_smoking_data()



mergings = linkage(df, method='complete')


# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels= smoking_valor,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()

# Kmeans
model = KMeans(n_clusters=4)

labels = model.fit_predict(df)

dataf = pd.DataFrame({'labels': labels, 'smoking': smoking_valor})

# Create crosstab: ct
ct = pd.crosstab(dataf['labels'], dataf['smoking'])

# Display ct
print(ct)

# Fcluster
labels1 = fcluster(mergings, 20000, criterion='distance')

dataf2 = pd.DataFrame({'labels': labels1, 'smoking':smoking_valor})

# Create crosstab: ct
ct2 = pd.crosstab(dataf2['labels'], dataf2['smoking'])

# Display ct
print(ct2)