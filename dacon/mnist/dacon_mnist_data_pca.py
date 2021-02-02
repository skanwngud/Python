import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

train=pd.read_csv('../data/dacon/data/train.csv', index_col=0, header=0)

x=train.iloc[:, 2:]

pca=PCA()
x=pca.fit_transform(x)

cumsum=np.cumsum(pca.explained_variance_ratio_)

print(np.argmax(cumsum>=0.99)+1) # 277