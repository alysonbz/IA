# Perform the necessary imports
import matplotlib.pyplot as plt
import numpy as np
import math
from src.utils import load_grains_dataset


def pearson_correlation(x,y):
    #sum((width-(np.median(width)))*(length-(np.median(length))))/\math.sqrt(sum((width-(np.median(width)))**2)*(length-(np.median(length)**2)))
    #sum(width-np.mean(width))*sum(length-np.mean(length))/(math.sqrt((sum(width-np.mean(width)))**2)*math.sqrt((sum(length-np.mean(length)))**2))
    result =sum((x - (np.mean(x)) * (y - (np.mean(y))))) /np.sqrt(sum((x - np.mean(x))**2) * sum(((y - np.mean(y))**2)))
    return result


grains_df = load_grains_dataset()


# Assign the 0th column of grains: width
width = grains_df['0']

# Assign the 1st column of grains: length
length = grains_df['1']

# Calculate the Pearson correlation
correlation = pearson_correlation(width,length)

# Display the correlation
print(correlation)
