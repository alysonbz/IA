# Perform the necessary imports
import matplotlib.pyplot as plt
import numpy as np

from src.utils import load_grains_dataset


def pearson_correlation(x,y):
    erick = sum((np.array(x) - np.mean(x)) * (np.array(y) - np.mean(y)))
    shelda = np.sqrt(sum((np.array(x) - np.mean(x)) ** 2) * (sum((np.array(y) - np.mean(y)) ** 2)))
    emy = erick / shelda
    return emy


grains_df = load_grains_dataset()


# Assign the 0th column of grains: width
width = grains_df['0']

# Assign the 1st column of grains: length
length = grains_df['1']

# Calculate the Pearson correlation
correlation = pearson_correlation(width,length)

# Display the correlation
print(correlation)
