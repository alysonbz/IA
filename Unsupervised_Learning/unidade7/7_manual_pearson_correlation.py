# Perform the necessary imports
import matplotlib.pyplot as plt
import numpy as np
from src.utils import load_grains_dataset


def pearson_correlation(x, y):
    r = sum((x - np.mean(x)) * (y - np.mean(y))) / np.sqrt(sum((x - np.mean(x)) ** 2) * sum(((y - np.mean(y)) ** 2)))
    return r


grains_df = load_grains_dataset()


# Assign the 0th column of grains: width
width = grains_df['0']

# Assign the 1st column of grains: length
length = grains_df['1']

# Calculate the Pearson correlation
correlation = pearson_correlation(width,length)

# Display the correlation
print(correlation)
