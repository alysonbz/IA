import matplotlib.pyplot as plt
from src.utils import load_grains_splited_datadet
import pandas as pd
#import linkage and dendogram
from scipy.cluster.hierarchy import linkage,dendrogram



X_train, samples, y_train, varieties = load_grains_splited_datadet()

# Calculate the linkage: mergings
mergings = linkage(samples, method= "complete")

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=8,
)
plt.show()