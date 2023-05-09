from src.utils import load_diabetes_clean_dataset
import matplotlib.pyplot as plt
from sklearn.linear_model import  LinearRegression


diabetes_df = load_diabetes_clean_dataset()
print(diabetes_df.head(5))

x = diabetes_df.drop("glucose", axis = 1).values
y = diabetes_df["glucose"].values
print(type(x), type(y))

x_bmi = x[:, 3]
print(y.shape, x_bmi.shape)
x_bmi = x_bmi.reshape(-1,  1)
print(x_bmi.shape)

plt.scatter(x_bmi, y)
plt.ylabel('Alguma coias')
plt.xlabel('algo')
plt.show()

#Plotanto a linha
reg = LinearRegression()
reg.fit(x_bmi, y)
predictions =  reg.predict(x_bmi)
plt.scatter(x_bmi, y)
plt.plot(x_bmi, predictions)
plt.ylabel("Blood Glucose (mg/dl)")
plt.xlabel("Body Rass  index")
plt.show()


