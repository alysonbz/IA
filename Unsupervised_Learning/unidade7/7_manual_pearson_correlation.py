# Perform the necessary imports
import matplotlib.pyplot as plt

from src.utils import load_grains_dataset
import numpy as np
import math

def pearson_correlation(x,y):
    xi = np.mean(x)
    yi = np.mean(y)
    formula = sum((x - xi)*(y-yi))/math.sqrt((sum((x-xi)**2))*(sum((y-yi)**2)))
    return formula


grains_df = load_grains_dataset()


# Assign the 0th column of grains: width
width = grains_df['0']

# Assign the 1st column of grains: length
length = grains_df['1']

# Calculate the Pearson correlation
correlation = pearson_correlation(width,length)

# Display the correlation
print(correlation)
