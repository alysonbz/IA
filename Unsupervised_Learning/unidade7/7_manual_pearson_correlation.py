# Perform the necessary imports
import matplotlib.pyplot as plt
import numpy as np

from src.utils import load_grains_dataset


def pearson_correlation(x,y):
    X_sub = (x - np.mean(x))
    Y_sub = (y - np.mean(y))
    r = sum(X_sub * Y_sub)
    r2 = r/np.sqrt((sum((X_sub)**2))*(sum((Y_sub)**2)))
    return r2


grains_df = load_grains_dataset()

# Assign the 0th column of grains: width
width = grains_df['0']

# Assign the 1st column of grains: length
length = grains_df['1']

# Calculate the Pearson correlation
correlation = pearson_correlation(width,length)

# Display the correlation
print(correlation)
