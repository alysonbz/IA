import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Carregar o dataset do arquivo CSV
lol_sample = pd.read_csv('lol_sample.csv')

# Carregar o dataset do arquivo CSV
lol_sample = pd.read_csv('C:/Users/eryka/OneDrive/Área de Trabalho/444/IA/AV3/lol_sample.csv')


# Definir as variáveis de entrada (X) e a variável de saída (y)
X = lol_sample.drop('blueWins', axis=1)
y = lol_sample['blueWins']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar uma instância do StandardScaler
scaler = StandardScaler()

# Ajustar o scaler aos dados de treinamento e normalizá-los
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Criar instâncias dos classificadores
svm = SVC()
rf = RandomForestClassifier()

# Treinar e avaliar o classificador SVM
svm.fit(X_train_normalized, y_train)
y_pred_svm = svm.predict(X_test_normalized)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Treinar e avaliar o classificador Random Forest
rf.fit(X_train_normalized, y_train)
y_pred_rf = rf.predict(X_test_normalized)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Comparar as acurácias dos classificadores
print("Acurácia SVM:", accuracy_svm)
print("Acurácia Random Forest:", accuracy_rf)
