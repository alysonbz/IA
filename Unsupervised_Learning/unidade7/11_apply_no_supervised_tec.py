from src.utils import load_fish_dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
fish_df = load_fish_dataset()


pca = PCA(n_components=2)
label_encoder = LabelEncoder()

specie = label_encoder.fit_transform(fish_df['specie'])

fish_df = fish_df.drop(['specie'], axis=1)

X = fish_df.values
y = fish_df.drop(['specie'].values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = LinearRegression()

regressor.fit(X_train, y_train)
