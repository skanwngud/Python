import numpy as np

from sklearn.datasets import load_iris

from sklearn.decomposition import PCA

datasets=load_iris()
x=datasets.data
y=datasets.target

pca=PCA()
pca.fit(x)

cumsum=np.cumsum(pca.explained_variance_ratio_)

print(np.argmax(cumsum>=0.98)+1)

# 3