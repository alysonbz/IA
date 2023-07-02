import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


cancer_df = pd.read_csv("breast-cancer.csv")

X = cancer_df.drop(['id', 'diagnosis'], axis=1)
y = cancer_df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

dt.fit(X_train, y_train_encoded)
rf.fit(X_train, y_train_encoded)


y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)


precision_dt = precision_score(y_test_encoded, y_pred_dt)
recall_dt = recall_score(y_test_encoded, y_pred_dt)

precision_rf = precision_score(y_test_encoded, y_pred_rf)
recall_rf = recall_score(y_test_encoded, y_pred_rf)

print("Decision Tree:")
print("Precisão:", precision_dt)
print("Recall:", recall_dt)

print("Random Forest:")
print("Precisão:", precision_rf)
print("Recall:", recall_rf)
