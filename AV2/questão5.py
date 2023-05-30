from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

#Gráfico da relação entre status de fumante e eficiência do sono
db = pd.read_csv("db_final.csv")
y = db["Sleep efficiency"].values
X = db["Smoking status"].values.reshape(-1, 1)

# Create the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X, y)

# Make predictions
predictions = reg.predict(X)

# Create scatter plot
plt.scatter(X, y, color="lightgreen")

# Create line plot
plt.plot(X, predictions, color="forestgreen")
plt.xlabel("Status de fumante")
plt.ylabel("Eficiência do sono")
plt.title('0-Fumante, 1-Não Fumante')

# Display the plot
plt.show()