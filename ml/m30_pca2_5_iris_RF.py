import numpy as np

from sklearn.datasets import load_iris

from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

datasets=load_iris()
x=datasets.data
y=datasets.target

print(x.shape)

for i in range(1, 5):
    pca=PCA(n_components=i)
    x2=pca.fit_transform(x)

    model=RandomForestRegressor()
    model.fit(x2,y)

    r2=model.score(x2, y)

    print(str(i) + ' 번째 r2 : ', r2)

# results
# 1 번째 r2 :  0.98367
# 2 번째 r2 :  0.992708
# 3 번째 r2 :  0.991554
# 4 번째 r2 :  0.991386