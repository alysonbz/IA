import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

test = pd.read_csv("test_dataset.csv")

test = test[['Area', 'Perimeter', 'ConvexArea', 'MajorAxisLength']]

X_train, samples, y_train, varieties = test


# Calculate the linkage: mergings
mergings = linkage(samples, method='complete')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()












