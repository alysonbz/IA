import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
smoke_detection = pd.read_csv("smoke_detection_iot.csv")
smoke = smoke_detection.drop(['Fire Alarm','Temperature[C]'],axis=1)
varieties = smoke_detection['Fire Alarm'].values

X_train, smoke, y_train, varieties = smoke_detection
mergings = linkage(smoke, method='complete')
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()

























"""




model = PCA()

pca_features = model.fit_transform(smoke)

xs = pca_features[:,0]

ys = pca_features[:,1]

plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

correlation, pvalue = pearsonr(xs, ys)

print(correlation)"""