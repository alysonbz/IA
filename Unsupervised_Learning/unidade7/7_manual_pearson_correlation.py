# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from src.utils import load_grains_dataset
import numpy as np

def pearson_correlation(x,y):
    r= sum((np.array(x) - np.mean(x))*(np.array(y) - np.mean(y)))
    t = np.sqrt()
    return None


grains_df = load_grains_dataset()


# Assign the 0th column of grains: width
width = grains_df['0']

# Assign the 1st column of grains: length
length = grains_df['1']

# Calculate the Pearson correlation
correlation = pearson_correlation(width,length)

# Display the correlation
print(correlation)
