from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.utils import load_fish_dataset
fish_df = load_fish_dataset()


# Faça uma redução da dimensão com PCA do dataset
# e escolha um método para classificar os atributos gerados pelo PCA obstidos na questão 10.
# Calcule a acurácia, gere o classification report e a matriz de confusão.

samples = fish_df.drop(['specie'],axis=1)
scaler = StandardScaler()
scaled_samples = scaler.fit_transform(samples)

pca = PCA(n_components=2)
pca.fit(scaled_samples)
transformed = pca.transform(scaled_samples)
print(transformed.shape)
