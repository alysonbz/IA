#bibliotecas necessárias
from src.utils import load_fish_dataset
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

#inicialização
pca = PCA()
knn = KNeighborsClassifier()
lb = LabelEncoder()
scaler = StandardScaler()

#carregador
fish_df = load_fish_dataset()
specie_encoded = lb.fit_transform(fish_df['specie'])
fish_df = fish_df.drop(['specie'],axis=1)
y = fish_df['specie'].values
scaled_fish = scaler.fit_transform(fish_df)

# redução da dimensão com PCA do dataset

pipeline = make_pipeline(scaler,pca)
pipeline.fit(fish_df)
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

#foi obtido o gráfico com as principais variancias da coluna especies, duas se sobressairam, com base nisso vamos análisar estes atributos, agora fica fácil de fazer essse processo, já que sabemos que apenas dois ficam acima de 1,5.
pca = PCA(n_components=2)
pca.fit(scaled_fish)
transformed = pca.transform(scaled_fish)
xs = transformed[:,0]
ys = transformed[:,1]
print(transformed.shape)
plt.scatter(xs,ys, c= specie_encoded)
plt.show()

#depois do processo do PCA, será escolhido um classificador, assim classificando os atributos que já foram gerados anteriormente, neste caso o classificador especifico usado foi o Knn.
#divisão em teste e treino
fish_df_train, fish_df_test, y_train, y_test = train_test_split(fish_df, y, stratify=y, random_state=42)
#ajustar o modelo(treino)
knn.fit(fish_df_train, y_train)
#acerto (teste)
print(knn.score(fish_df_test,y_test))
ac = str(round(knn.score(fish_df_test,y_test) * 100, 2))+"%"
print("A acurácia do modelo k-NN foi",ac)






