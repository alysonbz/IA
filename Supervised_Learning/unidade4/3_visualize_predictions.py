import matplotlib.pyplot as plt

from src.utils import processing_sales_clean
# Import matplotlib.pyplot


X,y,predictions = processing_sales_clean()

# Create scatter plot
plt.scatter(X,y, color="Blue")

# Create line plot
plt.plot(X, predictions, color="red")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

# Display the plot
plt.show()