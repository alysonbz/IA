import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler

ClanWaterQuality = pd.read_csv('dataset/CleanWaterQuality1.csv')

scaler = StandardScaler()
X = ClanWaterQuality[["aluminium","ammonia","arsenic","barium","cadmium","chloramine","chromium","copper",
                      "flouride","bacteria", "viruses","lead","nitrates","nitrites","mercury","perchlorate",
                      "radium","selenium","silver","uranium"]].values
y = ClanWaterQuality["is_safe"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)


transformer = FunctionTransformer(np.log1p)

X_train_norm = transformer.fit_transform(X_train)
X_test_norm = transformer.fit_transform(X_test)

dt_train = pd.DataFrame(X_train)
dt_test = pd.DataFrame(X_test)

X_train_scaled = pd.DataFrame(scaler.fit_transform(dt_train), columns=dt_train.columns)
X_test_scaled = pd.DataFrame(scaler.fit_transform(dt_test), columns = dt_test.columns)


knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)
knn.fit(X_train_norm, y_train)
knn.fit(X_train_scaled, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))

print(f"\n Sem sormalizção:{knn.score(X_test, y_test)}\n Normalização logarítmica:{knn.score(X_test_norm, y_test)}\n Normalização de media zero e variância unitária: {knn.score(X_test_scaled, y_test)}")