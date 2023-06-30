# Perform the necessary imports
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from src.utils import load_grains_dataset


grains_df = load_grains_dataset()


# Assign the 0th column of grains: width
width = grains_df['0']

# Assign the 1st column of grains: length
length = grains_df['1']

# Scatter plot width vs length
plt.scatter(width, length)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation
correlation, pvalue = pearsonr(width, length)
# Display the correlation
print(correlation)


