import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Carregar o dataset
sc = pd.read_csv(r"C:\Users\jonna\IA\AV1\dataset\star_classification_atualizado.csv")
print(sc.info())

sc_new = sc.dropna(subset=['class'])
label_encoders = {}
for column in sc_new.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    sc_new[column] = le.fit_transform(sc_new[column])
    label_encoders[column] = le

X = sc_new.drop('class', axis=1)
y = sc_new['class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
accuracies = {}

for metric in metrics:
    p = 2 if metric == 'minkowski' else 1
    knn_model = KNeighborsClassifier(n_neighbors=3, metric=metric, p=p)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_knn)
    accuracies[metric] = accuracy
    print(f'KNN Accuracy with {metric} distance: {accuracy:.4f}')

plt.figure(figsize=(10, 6))
bars = plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom')

plt.xlabel('Métrica de Distância')
plt.ylabel('Acurácia')
plt.title('Comparação das Acurácias com Diferentes Métricas de Distância no KNN')
plt.ylim(0, 1)
plt.show()