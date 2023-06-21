from src.utils import load_fish_dataset
fish_df = load_fish_dataset()
from sklearn.decomposition import PCA

pca = PCA()
