# Perform the necessary imports
import numpy as np
import matplotlib.pyplot as plt

from src.utils import load_grains_dataset


def pearson_correlation(x,y):
    x_bar= np.mean(x)
    y_bar= np.mean(y)
    somatorio = sum((x-x_bar) * (y-y_bar))
    somatorio2 = sum(((x-x_bar)**2)) * sum((y-y_bar)**2)
    div = somatorio / (np.sqrt(somatorio2))
    return div



grains_df = load_grains_dataset()


# Assign the 0th column of grains: width
width = grains_df["0"]

# Assign the 1st column of grains: length
length = grains_df["1"]

# Calculate the Pearson correlation
correlation = pearson_correlation(width,length)

# Display the correlation
print(correlation)
