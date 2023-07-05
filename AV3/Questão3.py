import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carregar o dataset
df = pd.read_csv("mitbih_train.csv")

# Pré-processamento dos dados
X = df.drop('9.779411554336547852e-01', axis=1)
y = pd.cut(df['9.779411554336547852e-01'], bins=[-float('inf'), 0.5, float('inf')], labels=['Normal', 'Abnormal'])

# Divisão do conjunto de dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicação do classificador (Random Forest)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Avaliação do desempenho
accuracy = accuracy_score(y_test, y_pred)

# Resultado
print("Acurácia:", accuracy)
