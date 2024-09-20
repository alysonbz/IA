import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r'C:\\Users\\jonna\\IA\\AV2\\dataset\\carprice.csv') 

df.replace('?', np.nan, inplace=True)

df['curb-weight'] = pd.to_numeric(df['curb-weight'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

df.dropna(subset=['curb-weight', 'price'], inplace=True)

X = df[['curb-weight']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rss = np.sum((y_test - y_pred) ** 2)
mse = mean_squared_error(y_test, y_pred) 
rmse = np.sqrt(mse)
r_squared = r2_score(y_test, y_pred)

# Gráfico da reta de regressão
plt.scatter(X, y, color='blue', label='Dados Reais')
plt.plot(X_test, y_pred, color='red', label='Reta de Regressão')
plt.xlabel('Curb Weight')
plt.ylabel('Price')
plt.title('Regressão Linear: Price vs Curb Weight')
plt.legend()
plt.show()

# Exibir as métricas
print(f'RSS: {rss}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R²: {r_squared}')