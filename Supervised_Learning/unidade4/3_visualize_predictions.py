from src.utils import processing_sales_clean
#Import matplotlib.pyplot
import matplotlib.pyplot as plt

X,y,predictions = processing_sales_clean()

# Create scatter plot
plt.scatter(X, y, color="green")

# Create line plot
plt.plot(X, predictions, color="purple")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

# Display the plot
plt.show()