from sklearn.datasets import load_wine

from sklearn.decomposition import PCA

import numpy as np

datasets=load_wine()
x=datasets.data
y=datasets.target

pca=PCA()
pca.fit(x)
cumsum=np.cumsum(pca.explained_variance_ratio_)

print(np.argmax(cumsum>=0.999)+1)

# 2