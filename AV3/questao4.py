import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

glass_df = pd.read_csv("glass.csv")

X = glass_df.drop(['Type'], axis=1)
y = glass_df['Type']

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

precision_dt = precision_score(y_test_encoded, y_pred_dt, average='weighted')
recall_dt = recall_score(y_test_encoded, y_pred_dt, average='weighted')
f1_dt = f1_score(y_test_encoded, y_pred_dt, average='weighted')
cm_dt = confusion_matrix(y_test_encoded, y_pred_dt)

precision_rf = precision_score(y_test_encoded, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test_encoded, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test_encoded, y_pred_rf, average='weighted')
cm_rf = confusion_matrix(y_test_encoded, y_pred_rf)

print("Decision Tree:")
print("Precision:", precision_dt)
print("Recall:", recall_dt)
print("F1-score:", f1_dt)
print("Confusion Matrix:")
print(cm_dt)

print("Random Forest:")
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1-score:", f1_rf)
print("Confusion Matrix:")
print(cm_rf)
